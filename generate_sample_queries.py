import sqlite3
from typing import Iterator, List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass, asdict
import asyncio
import litellm
import json
from local_email_db import DEFAULT_DB_PATH
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from utils import cache
from test_and_train_inboxes import train_inboxes, test_inboxes
from datasets import Dataset, Features, Value, Sequence
from huggingface_hub import HfApi, create_repo
from panza import limit_concurrency
from tqdm.asyncio import tqdm
from types_enron import SyntheticQuery
import random
from datetime import datetime, timedelta
from litellm.caching.caching import LiteLLMCacheType, Cache

litellm.cache = Cache(type=LiteLLMCacheType.DISK)

load_dotenv()


# Configure logging to output to both console and a file
log_file_path = "sample_queries.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(),  # Keep console output as well
    ],
)


# Define a dataclass for email data
@dataclass
class EmailData:
    id: int
    message_id: str
    subject: str
    from_address: str
    to_address: list[str]
    cc_address: list[str]
    bcc_address: list[str]
    date: str
    body: str


def iterate_inbox_emails_by_date(
    inbox_address: str, db_path: str = DEFAULT_DB_PATH, chunk_size: int = 10
) -> Iterator[List[EmailData]]:  # Update return type annotation
    """
    Iterates over all emails associated with a specific inbox address,
    ordered by date (oldest to newest), yielding them in chunks.

    Filters emails to include only those with body length between 200 and 5000 characters
    and dated 1995 or later.

    An email is considered associated with the inbox if the address is
    in the 'from', 'to', 'cc', or 'bcc' fields.

    Args:
        inbox_address: The email address of the inbox to retrieve emails for.
        db_path: The path to the SQLite database file.
        chunk_size: The number of emails to yield in each chunk.

    Yields:
        A list of EmailData objects, where each object represents an email.
        Each list represents a chunk of emails.
    """
    logging.info(
        f"Iterating emails for inbox '{inbox_address}' from '{db_path}' in chunks of {chunk_size}"
    )
    conn: Optional[sqlite3.Connection] = None
    cursor: Optional[sqlite3.Cursor] = None  # Initialize cursor to None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query to find all unique email IDs associated with the inbox address
        # This involves checking both the sender and recipients tables
        # Using UNION ensures we get distinct email IDs
        # Added filters for body length and date
        # Use subqueries with GROUP_CONCAT to fetch recipient lists
        query = """
        SELECT
            e.id,
            e.message_id,
            e.subject,
            e.from_address,
            -- Aggregate 'to' addresses
            (SELECT GROUP_CONCAT(r.recipient_address) FROM recipients r WHERE r.email_id = e.id AND r.recipient_type = 'to') AS to_addresses,
            -- Aggregate 'cc' addresses
            (SELECT GROUP_CONCAT(r.recipient_address) FROM recipients r WHERE r.email_id = e.id AND r.recipient_type = 'cc') AS cc_addresses,
            -- Aggregate 'bcc' addresses
            (SELECT GROUP_CONCAT(r.recipient_address) FROM recipients r WHERE r.email_id = e.id AND r.recipient_type = 'bcc') AS bcc_addresses,
            e.date,
            e.body
        FROM emails e
        WHERE
            e.id IN ( -- Select emails where the inbox address is involved
                SELECT e_sub.id
                FROM emails e_sub
                WHERE e_sub.from_address = :address
                UNION -- Use UNION to combine with emails where the address is a recipient
                SELECT r_sub.email_id
                FROM recipients r_sub
                WHERE r_sub.recipient_address = :address
            )
            AND LENGTH(e.body) >= 200
            AND LENGTH(e.body) <= 5000
            AND STRFTIME('%Y', e.date) >= '1995'
        ORDER BY e.date ASC;
        """

        cursor.execute(query, {"address": inbox_address})

        while True:
            chunk_rows = cursor.fetchmany(chunk_size)
            if not chunk_rows:
                break

            # Convert rows to EmailData objects, splitting recipient strings
            chunk_data = []
            for row in chunk_rows:
                # Unpack the row based on the new query structure
                (
                    id_val,
                    message_id,
                    subject,
                    from_address,
                    to_str,
                    cc_str,
                    bcc_str,
                    date,
                    body,
                ) = row

                # Split comma-separated strings into lists, handling None/NULL
                to_list = to_str.split(",") if to_str else []
                cc_list = cc_str.split(",") if cc_str else []
                bcc_list = bcc_str.split(",") if bcc_str else []

                chunk_data.append(
                    EmailData(
                        id=id_val,
                        message_id=message_id,
                        subject=subject,
                        from_address=from_address,
                        to_address=to_list,
                        cc_address=cc_list,
                        bcc_address=bcc_list,
                        date=date,
                        body=body,
                    )
                )
            yield chunk_data

        logging.info(f"Finished iterating emails for inbox '{inbox_address}'")

    except sqlite3.Error as e:
        logging.error(f"Database error while accessing inbox '{inbox_address}': {e}")
        # Depending on requirements, you might want to raise the exception
        # or yield an empty list / handle differently
        yield []  # Yield empty list on error to signal failure gracefully
    finally:
        # Ensure cursor and conn are closed even if connection fails before cursor assignment
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            logging.debug(f"Database connection closed for inbox '{inbox_address}'.")


system_prompt = """We are training an email assistant. The user will query their email inbox in natural language, and the assistant will find relevant emails and answer the user's questions.

Your job is to generate synthetic training data for the assistant. You will be passed 20 emails, and you need to generate plausible sample questions that a user might ask the agent, whose answers are all contained in the emails. The questions should be short and to the point, and have concrete answers present in the emails. For each question, you should also return the correct answer as well as the email ids (the integer primary key `id` field, NOT the `message_id` string) of the emails that contain the answer. Note that some batches of emails might not be good candidates for generating training data. In that case you can return an empty list. The user's email inbox is {inbox_address}.

- Questions should be in the first person on behalf of the user. Eg. 'What quote did John give me for project X?'
- The questions should be short and to the point, and have concrete answers present in the emails.
- Try to imagine questions that a real person would ask about the emails, with just the level of detail they'd likely remember.
- Just use first names, not full names in the questions. Eg. 'What quote did James give me for project X?', not 'What quote did James Wong give me for project X?'
- Return only a JSON list of objects. Each object should have the following fields:
  - question: string, (The question the user might ask)
  - answer: string, (The answer to the question)
  - email_ids: int[], (The integer primary key `id`s of the emails that contain the answer)
  - how_realistic: float, (How likely a user would be to actually ask this question, between 0 and 1)
"""


# Use Pydantic for validation from LLM response
class LLMSampleQuery(BaseModel):
    question: str
    answer: str
    email_ids: List[int]  # Expecting integer PK IDs from LLM
    how_realistic: float

    # Define the structure for the final dataset

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


@limit_concurrency(25)
async def generate_queries_for_batch(
    inbox_address: str,
    email_batch: List[EmailData],
    system_prompt_template: str,
    model_id: str,
) -> List[SyntheticQuery]:
    """
    Generates sample queries for a batch of emails using an LLM,
    maps email PK IDs to message_ids, and adds inbox_address.
    """
    logging.info(
        f"Generating queries for batch for inbox {inbox_address} using model: {model_id}"
    )

    # Create a mapping from email PK ID to message_id for this batch
    id_to_message_id_map = {email.id: email.message_id for email in email_batch}
    # Create a mapping from email PK ID to email date
    id_to_date = {email.id: email.date for email in email_batch}

    # Format the system prompt with the specific inbox address
    formatted_system_prompt = system_prompt_template.format(inbox_address=inbox_address)

    # Format the email batch for the user message
    email_batch_dicts = [asdict(email) for email in email_batch]
    email_batch_json_str = json.dumps(email_batch_dicts, indent=2)

    try:
        response = await litellm.acompletion(
            model=model_id,
            messages=[
                {"role": "system", "content": formatted_system_prompt},
                {"role": "user", "content": email_batch_json_str},
            ],
            max_completion_tokens=10000,  # Adjust as needed
            temperature=0.5,  # Add some temperature for variety
            caching=True,
        )
        raw_content: str = response.choices[0].message.content  # type: ignore
    except Exception as e:
        logging.error(f"LLM completion failed for inbox {inbox_address}: {e}")
        return []  # Return empty list on LLM error

    processed_queries: List[SyntheticQuery] = []
    try:
        # Extract content between the first [ and last ]
        first_bracket = raw_content.find("[")
        last_bracket = raw_content.rfind("]")
        if first_bracket != -1 and last_bracket != -1:
            response_content_str = raw_content[first_bracket : last_bracket + 1]
        else:
            if raw_content.strip().startswith("{"):
                response_content_str = (
                    f"[{raw_content.strip()}]"  # Wrap single object in list
                )
            else:
                response_content_str = raw_content  # Fallback
            logging.warning(
                f"Could not find JSON list brackets in response for {inbox_address}. Attempting to parse anyway. Raw content:\n{raw_content}"
            )

        response_data = json.loads(response_content_str)
        if not isinstance(response_data, list):
            logging.error(
                f"LLM response was not a JSON list for inbox {inbox_address}. Content: {response_content_str}"
            )
            return []

        llm_queries = [LLMSampleQuery.model_validate(query) for query in response_data]

        # Convert LLM queries to Dataset queries (map IDs, add inbox)
        for llm_query in llm_queries:
            message_ids = []
            valid_ids = True
            for email_pk_id in llm_query.email_ids:
                msg_id = id_to_message_id_map.get(email_pk_id)
                if msg_id:
                    message_ids.append(msg_id)
                else:
                    logging.warning(
                        f"LLM returned email_id {email_pk_id} which is not in the current batch for inbox {inbox_address}. Skipping this query."
                    )
                    valid_ids = False
                    break  # Skip this query if any ID is invalid
            if valid_ids:
                email_dates = [
                    datetime.fromisoformat(id_to_date[email_pk_id])
                    for email_pk_id in llm_query.email_ids
                    if email_pk_id in id_to_date
                ]
                latest_date = max(email_dates) if email_dates else datetime.now()
                days_after = random.randint(1, 30)
                rand_seconds = random.randint(0, 86400)
                query_date = latest_date + timedelta(
                    days=days_after, seconds=rand_seconds
                )
                query_date_str = query_date.strftime("%Y-%m-%d")

                processed_queries.append(
                    SyntheticQuery(
                        question=llm_query.question,
                        answer=llm_query.answer,
                        message_ids=message_ids,
                        how_realistic=llm_query.how_realistic,
                        inbox_address=inbox_address,
                        query_date=query_date_str,  # New field added
                    )
                )

    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        logging.error(
            f"Error parsing LLM response content for inbox {inbox_address}: {e}"
        )
        logging.error(f'Raw LLM response content:\n"""\n{raw_content}\n"""\n')
        return []
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during query processing for {inbox_address}: {e}"
        )
        logging.error(f'Raw LLM response content:\n"""\n{raw_content}\n"""\n')
        return []

    logging.info(
        f"Successfully generated and processed {len(processed_queries)} queries for batch for inbox {inbox_address}."
    )
    return processed_queries


async def create_and_push_dataset(
    num_train_inboxes: int,
    num_test_inboxes: int,
    batches_per_inbox: int = 5,
    db_path: str = DEFAULT_DB_PATH,
    chunk_size: int = 20,
    model_id: str = "gemini/gemini-2.5-pro-preview-03-25",
    hf_repo_id: str = "corbt/enron_emails_sample_questions",
):
    """
    Generates sample queries concurrently for specified numbers of train and test inboxes,
    creates a Hugging Face dataset, and pushes it to the Hub.
    """
    logging.info(f"Starting dataset generation for {hf_repo_id}")
    logging.info(
        f"Processing {num_train_inboxes} train inboxes and {num_test_inboxes} test inboxes, "
        f"up to {batches_per_inbox} batches each."
    )
    logging.info(f"Using model: {model_id}")
    # await generate_queries_for_batch.bust_cache()

    # Select inboxes from both train and test sets
    selected_train_inboxes = train_inboxes[:num_train_inboxes]
    selected_test_inboxes = test_inboxes[:num_test_inboxes]
    target_inboxes = selected_train_inboxes + selected_test_inboxes
    logging.info(f"Total target inboxes: {len(target_inboxes)}")

    if not target_inboxes:
        logging.warning("No target inboxes selected. Exiting.")
        return

    tasks = []
    total_batches_to_process = 0

    # --- Task Creation Phase ---
    logging.info("Creating generation tasks...")
    for inbox_address in target_inboxes:
        logging.info(f"--- Preparing tasks for inbox: {inbox_address} ---")
        try:
            email_chunk_iterator = iterate_inbox_emails_by_date(
                inbox_address, db_path=db_path, chunk_size=chunk_size
            )
            batches_prepared_for_inbox = 0
            for i in range(batches_per_inbox):
                logging.debug(
                    f"Fetching batch {i+1}/{batches_per_inbox} for {inbox_address}..."
                )
                try:
                    email_chunk = next(email_chunk_iterator)
                    if not email_chunk:
                        logging.info(
                            f"No more emails found for {inbox_address} after batch {i}. Stopping task creation for this inbox."
                        )
                        break  # No more emails for this inbox

                    # Create the coroutine and add it to the list
                    task = generate_queries_for_batch(
                        inbox_address=inbox_address,
                        email_batch=email_chunk,
                        system_prompt_template=system_prompt,
                        model_id=model_id,
                    )
                    tasks.append(task)
                    batches_prepared_for_inbox += 1
                    total_batches_to_process += 1

                except StopIteration:
                    logging.info(
                        f"StopIteration: No more batches available for {inbox_address} after batch {i}. Stopping task creation for this inbox."
                    )
                    break  # Exhausted iterator for this inbox
                except Exception as e:
                    # Log error during batch fetching but continue if possible
                    logging.error(
                        f"An unexpected error occurred fetching batch {i+1} for {inbox_address}: {e}",
                        exc_info=True,
                    )
                    continue  # Try next batch for this inbox

            logging.info(
                f"--- Prepared {batches_prepared_for_inbox} tasks for inbox {inbox_address} ---"
            )
        except Exception as e:
            # Log error initializing iterator or other inbox-level issues
            logging.error(
                f"Failed to prepare tasks for inbox {inbox_address}: {e}", exc_info=True
            )
            continue  # Move to the next inbox

    logging.info(f"Total generation tasks created: {len(tasks)}")
    if not tasks:
        logging.warning("No tasks were created. Exiting.")
        return

    # --- Task Execution Phase ---
    logging.info(f"Running {len(tasks)} generation tasks concurrently...")
    results = await tqdm.gather(*tasks)
    logging.info("All generation tasks completed.")

    # --- Result Aggregation Phase ---
    all_queries: List[SyntheticQuery] = []
    successful_batches = 0
    failed_batches = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Log the exception details - you might want more specific logging
            # based on which task failed, potentially linking back to inbox/batch index if needed
            logging.error(f"Task {i} failed: {result}", exc_info=result)
            failed_batches += 1
        elif isinstance(result, list):
            # Successfully received a list (potentially empty) of queries
            all_queries.extend(result)
            successful_batches += 1
            # Optional: Log success per task if needed
            # logging.debug(f"Task {i} succeeded, added {len(result)} queries.")
        else:
            # Handle unexpected return types if necessary
            logging.error(f"Task {i} returned unexpected type: {type(result)}")
            failed_batches += 1

    logging.info(f"--- Aggregation Summary ---")
    logging.info(
        f"Total batches processed: {successful_batches + failed_batches} (Target: {total_batches_to_process})"
    )
    logging.info(f"Successful batches: {successful_batches}")
    logging.info(f"Failed batches: {failed_batches}")
    logging.info(
        f"Total queries generated across all successful batches: {len(all_queries)}"
    )

    if not all_queries:
        logging.warning(
            "No queries were successfully generated. Skipping dataset creation and push."
        )
        return

    # --- Create Hugging Face Dataset ---
    logging.info("Preparing data for Hugging Face dataset...")

    # Convert list of Pydantic objects to dict of lists (now including query_date)
    data_dict = {
        "question": [q.question for q in all_queries],
        "answer": [q.answer for q in all_queries],
        "message_ids": [q.message_ids for q in all_queries],
        "how_realistic": [q.how_realistic for q in all_queries],
        "inbox_address": [q.inbox_address for q in all_queries],
        "query_date": [q.query_date for q in all_queries],  # New field
    }

    # Define dataset features (adding query_date as a string type)
    features = Features(
        {
            "question": Value("string"),
            "answer": Value("string"),
            "message_ids": Sequence(Value("string")),
            "how_realistic": Value("float32"),
            "inbox_address": Value("string"),
            "query_date": Value("string"),  # New field
        }
    )

    try:
        logging.info("Creating dataset object...")
        hf_dataset = Dataset.from_dict(data_dict, features=features)
        logging.info(f"Dataset created with {len(hf_dataset)} rows.")

        # --- Push to Hub ---
        logging.info(f"Pushing dataset to Hugging Face Hub: {hf_repo_id}")
        # Ensure the repo exists, create if it doesn't (make it public)
        create_repo(hf_repo_id, repo_type="dataset", exist_ok=True)

        hf_dataset.push_to_hub(hf_repo_id, private=False)  # Ensure public
        logging.info(f"Successfully pushed dataset to {hf_repo_id}")
        print(
            f"\nDataset successfully pushed: https://huggingface.co/datasets/{hf_repo_id}"
        )

    except Exception as e:
        logging.error(
            f"Failed to create or push dataset to Hugging Face Hub: {e}", exc_info=True
        )
        print(
            "\nFailed to create or push dataset. Check logs and Hugging Face authentication (`huggingface-cli login`)."
        )


# --- Main Execution ---
if __name__ == "__main__":
    # Example: Generate queries for the first 2 train inboxes and 1 test inbox,
    # 5 batches each, and push to corbt/enron_emails_sample_questions
    # Ensure you are logged in: `huggingface-cli login`
    asyncio.run(
        create_and_push_dataset(
            num_train_inboxes=20,
            num_test_inboxes=20,
            batches_per_inbox=5,
            hf_repo_id="corbt/enron_emails_sample_questions",
        )
    )
