import logging
from pathlib import Path
from local_email_db import DEFAULT_DB_PATH, generate_database
from email_search_tools import EmailSearchTools

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Example usage of EmailSearchTools."""
    # Ensure the database exists first
    logging.info("Checking/Generating database...")
    db_path = Path(DEFAULT_DB_PATH)
    if not db_path.exists():
        logging.warning(f"Database not found at {db_path}. Running generation.")
        generate_database()  # Generate if missing
    logging.info("Database ready.")

    # Create an instance of EmailSearchTools
    email_tools = EmailSearchTools()

    # --- Example Usage ---
    test_inbox = "phillip.allen@enron.com"  # A known user in the dataset
    logging.info(f"\n--- Running Search Examples for Inbox: {test_inbox} ---")

    print("\n--- Search Example 1: Keywords 'forecast', 'gas' ---")
    search_results_1 = email_tools.search_emails(
        inbox=test_inbox, keywords=["forecast", "gas"], max_results=5
    )
    print(f"Found {len(search_results_1)} results:")
    first_email_id_to_get = None
    for i, result in enumerate(search_results_1):
        print(f"  Email ID: {result.message_id}, Snippet: {result.snippet}")
        if i == 0:  # Store the first ID found
            first_email_id_to_get = result.message_id
    print("----------------------------------------")

    print("\n--- Search Example 2: Keyword 'meeting', From specific sender ---")
    test_sender = "pallen70@aol.com"
    search_results_2 = email_tools.search_emails(
        inbox=test_inbox, keywords=["meeting"], from_addr=test_sender, max_results=5
    )
    print(f"Found {len(search_results_2)} results (from {test_sender}):")
    for result in search_results_2:
        print(f"  Email ID: {result.message_id}, Snippet: {result.snippet}")
    print("----------------------------------------")

    # --- Example Usage for read_email ---
    print("\n--- Get Email Example ---")
    if first_email_id_to_get:
        print(f"Attempting to retrieve email with ID: {first_email_id_to_get}")
        retrieved_email = email_tools.read_email(first_email_id_to_get)
        if retrieved_email:
            print("Successfully retrieved email:")
            print(f"  Message ID: {retrieved_email.message_id}")
            print(f"  Date: {retrieved_email.date}")
            print(f"  From: {retrieved_email.from_address}")
            print(f"  To: {retrieved_email.to_addresses}")
            print(f"  Cc: {retrieved_email.cc_addresses}")
            print(f"  Subject: {retrieved_email.subject}")
            # Avoid printing the full body, it can be long
            body_preview = (
                (retrieved_email.body or "")[:100] + "..."
                if retrieved_email.body
                else "N/A"
            )
            print(f"  Body Preview: {body_preview}")
        else:
            print(f"Could not retrieve email with ID: {first_email_id_to_get}")
    else:
        print(
            "Skipping read_email example as no email ID was found in Search Example 1."
        )
    print("----------------------------------------")


if __name__ == "__main__":
    # Set logging level to DEBUG to see SQL queries if needed
    # logging.getLogger().setLevel(logging.DEBUG)
    main()
