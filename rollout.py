from typing import List, Any
from types_enron import SyntheticQuery
from art import Trajectory
from art.types import MessagesAndChoices
from openai.types.chat.chat_completion import Choice
from litellm import acompletion
import litellm
from email_search_tools import search_emails, read_email
from langchain_core.utils.function_calling import convert_to_openai_tool
from litellm.caching.caching import LiteLLMCacheType, Cache
import json
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from litellm.types.utils import Choices, ModelResponse
from dataclasses import asdict
from utils import convert_litellm_choice_to_openai
from dataclasses import dataclass
from art.utils import limit_concurrency
from tqdm.asyncio import tqdm
import os
from openpipe import AsyncOpenPipe
from datetime import datetime


litellm.cache = Cache(type=LiteLLMCacheType.DISK)

# Initialize OpenPipe client (ensure OPENPIPE_API_KEY is in your .env)
if os.getenv("OPENPIPE_API_KEY"):
    op_client = AsyncOpenPipe()
else:
    op_client = None

"""
Steps for implementing the rollout function:

"""

# We remove the inbox parameter before describing the tool for OpenAI because we'll set that value explicitly based on the user we're running on behalf of.
search_tool = convert_to_openai_tool(search_emails)
del search_tool["function"]["parameters"]["properties"]["inbox"]
search_tool["function"]["parameters"]["required"].remove("inbox")

MAX_TURNS = 10


@dataclass
class FinalRubric:
    cant_parse_tool_call: bool = False
    bad_tool_call_name: bool = False
    bad_tool_call_args: bool = False
    ran_out_of_turns: bool = False
    returned_i_dont_know: bool = False
    attempted_answer: bool = False
    answer_correct: bool = False
    sources_correct: bool = False
    num_sources: int = 0
    num_turns: int = 0
    ever_found_right_email: bool = False
    ever_read_right_email: bool = False
    ever_tried_to_read_invalid_email: bool = False

    def to_metrics(self) -> dict[str, float | int]:
        return {k: int(v) for k, v in asdict(self).items()}


def reward_and_metrics(
    trajectory: Trajectory, rubric: FinalRubric
) -> tuple[float, dict]:
    metrics = rubric.to_metrics()

    # Note: make sure all possible partial rewards always sum to less than 0.5.
    partial_rewards = 0
    partial_rewards += (
        (rubric.num_turns / MAX_TURNS) / 10
    )  # You get up to 0.1 points for running through the turn limit instead of giving up early
    partial_rewards += 0.1 if rubric.ever_found_right_email else 0
    partial_rewards += 0.1 if rubric.ever_read_right_email else 0
    partial_rewards += 0.1 if not rubric.ever_tried_to_read_invalid_email else 0
    partial_rewards += 0.1 if rubric.sources_correct else 0

    # Formatting error: reward will be -2 to -1
    if rubric.cant_parse_tool_call:
        return -2 + partial_rewards, metrics

    if rubric.bad_tool_call_name:
        return -1.9 + partial_rewards, metrics

    if rubric.bad_tool_call_args:
        return -1.8 + partial_rewards, metrics

    # No formatting error, but wrong answer: reward will be -1 to 0
    if rubric.attempted_answer and not rubric.answer_correct:
        return -1 + partial_rewards, metrics

    # Returned no answer at all: reward will be 0 to 1
    if rubric.returned_i_dont_know or rubric.ran_out_of_turns:
        return 0 + partial_rewards, metrics

    # Answer is correct: reward will be 1 to 2
    if rubric.answer_correct:
        # Partial credit calculation is different for correct answers.

        reward = 1
        reward += 0.3 if rubric.sources_correct else 0

        # Extra credit for not including extra sources.
        reward += 0.1 / rubric.num_sources if rubric.num_sources > 0 else 0

        # Extra credit for being faster.
        reward += 0.1 * rubric.num_turns / MAX_TURNS
        return reward, metrics

    print(rubric)
    raise ValueError("Rubric is not handled properly")


def return_final_answer(answer: str, sources: List[str] | None) -> str:
    """
    This function is used to return the final answer to the user's query.
    It should be called with the answer and the sources. If you cannot find the answer, you should return "I don't know" with an empty list of sources.

    Args:
        answer: (str) the answer to the user's query. If you cannot find the answer, you should return "I don't know" with an empty list of sources.
        sources: (list[str]) a list of message ids that are relevant to the query. Usually there will be only one.

    Returns:
        (str) the final answer to the user's query
    """
    ...


def tool_response(response: Any) -> ChatCompletionMessageParam:
    return {
        "role": "user",
        "content": json.dumps(response),
    }


tools: list[ChatCompletionToolParam] = [
    search_tool,
    convert_to_openai_tool(read_email),
    convert_to_openai_tool(return_final_answer),
]  # type: ignore


def to_messages(
    messages_and_choices: MessagesAndChoices,
) -> list[ChatCompletionMessageParam]:
    messages = []
    for message in messages_and_choices:
        if isinstance(message, Choice):
            messages.append(message.message.to_dict())
        else:
            messages.append(message)
    return messages


async def determine_if_answer_is_correct(answer: str, query: SyntheticQuery) -> bool:
    system_prompt = "You will be given an question and two different answers to the question, the correct answer and the answer given by an AI. Your job is to determine if the answer given by the AI is correct. Return True if the answer is semantically similar to the correct answer, and False otherwise. Return only the word True or False, no other text."

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Question: {query.question}\nCorrect answer: {query.answer}\nAI answer: {answer}",
        },
    ]

    response = await acompletion(
        model="gemini/gemini-2.0-flash",
        messages=messages,
        temperature=0,
        caching=True,
        max_tokens=2,
    )

    return response.choices[0].message.content.strip().lower().startswith("t")  # type: ignore


@limit_concurrency(10)
async def rollout(
    model: str,
    scenario: SyntheticQuery,
    base_url: str | None = None,
    api_key: str | None = None,
    trainable: bool = False,
    log_to_openpipe: bool = True,
) -> Trajectory:
    rollout_start_time = datetime.now()  # Record start time
    rubric = FinalRubric()
    traj = Trajectory(metadata={"email_inbox": scenario.inbox_address})

    system_prompt = f"""You are an email search agent. You are given a user query and a list of tools you can use to search the user's email. Use the tools to search the user's emails and find the answer to the user's query. You may take up to {MAX_TURNS} turns to find the answer, so if your first seach doesn't find the answer, you can try with different keywords.
    
    Here are the tools you can use:
    {tools}
    
    Respond with a valid JSON object with the following fields:
    - tool_name: (str) the name of the tool to use
    - tool_args: (JSON) the arguments to pass to the tool

    For example, to read a specific email, you should respond with:
    {{
        "tool_name": "read_email",
        "tool_args": {{
            "message_id": "<12635597.1075855702772.JavaMail.evans@thyme>"
        }}
    }}

    User's email address is {scenario.inbox_address}
    Today's date is {scenario.query_date}
    """
    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario.question},
    ]
    llm_response: ModelResponse | None = None
    final_answer = None

    while True:
        rubric.num_turns += 1

        if rubric.num_turns > MAX_TURNS:
            rubric.ran_out_of_turns = True
            break

        llm_response = await acompletion(
            model=model,
            base_url=base_url,
            messages=to_messages(traj.messages_and_choices),
            caching=not trainable,
            api_key=api_key,
        )  # type: ignore

        choice = llm_response.choices[0]  # type: ignore
        assert isinstance(choice, Choices)
        if trainable:
            traj.messages_and_choices.append(convert_litellm_choice_to_openai(choice))
        else:
            traj.messages_and_choices.append(choice.message.to_dict())  # type: ignore
        raw_content = choice.message.content
        assert raw_content is not None
        start_index = raw_content.find("{")
        end_index = raw_content.rfind("}")
        if not (start_index != -1 and end_index != -1 and start_index < end_index):
            rubric.cant_parse_tool_call = True
            break
        json_str = raw_content[start_index : end_index + 1]

        try:
            tool_call = json.loads(json_str)
        except Exception as e:
            traj.logs.append(f"Error parsing tool call: {e}")
            rubric.cant_parse_tool_call = True
            break

        if "tool_args" not in tool_call:
            rubric.bad_tool_call_args = True
            traj.logs.append(f"Tool call missing tool_args: {tool_call}")
            break

        match tool_call.get("tool_name"):
            case "search_emails":
                try:
                    search_results = search_emails(
                        inbox=scenario.inbox_address,
                        keywords=tool_call["tool_args"].get("keywords"),
                        from_addr=tool_call["tool_args"].get("from_addr"),
                        to_addr=tool_call["tool_args"].get("to_addr"),
                        sent_after=tool_call["tool_args"].get("sent_after"),
                        sent_before=tool_call["tool_args"].get("sent_before"),
                        max_results=tool_call["tool_args"].get("max_results"),
                    )
                    traj.messages_and_choices.append(
                        tool_response([asdict(r) for r in search_results])
                    )
                    for r in search_results:
                        if r.message_id == scenario.message_ids[0]:
                            rubric.ever_found_right_email = True
                except Exception as e:
                    rubric.bad_tool_call_args = True
                    traj.logs.append(f"Error searching emails: {e}")
                    break
            case "read_email":
                message_id_to_read = tool_call["tool_args"].get("message_id")
                if message_id_to_read == scenario.message_ids[0]:
                    rubric.ever_read_right_email = True
                email_content = read_email(message_id_to_read)
                if email_content is None:
                    traj.messages_and_choices.append(
                        tool_response({"error": "Email not found"})
                    )
                    rubric.ever_tried_to_read_invalid_email = True
                else:
                    traj.messages_and_choices.append(
                        tool_response(email_content.model_dump())
                    )
            case "return_final_answer":
                final_answer = tool_call["tool_args"]["answer"]
                final_sources = tool_call["tool_args"]["sources"]

                if final_answer == "I don't know":
                    rubric.returned_i_dont_know = True
                else:
                    rubric.attempted_answer = True
                    rubric.answer_correct = await determine_if_answer_is_correct(
                        final_answer, scenario
                    )
                    rubric.sources_correct = scenario.message_ids[0] in final_sources
                break
            case _:
                rubric.bad_tool_call_name = True
                break

    reward, metrics = reward_and_metrics(traj, rubric)
    traj.reward = reward
    traj.metrics = metrics
    rollout_end_time = datetime.now()  # Record end time

    if log_to_openpipe and op_client is not None:
        try:
            await op_client.report(
                requested_at=rollout_start_time.timestamp() * 1000,
                received_at=rollout_end_time.timestamp() * 1000,
                req_payload={
                    "model": model,
                    "messages": to_messages(traj.messages_and_choices)[:-1],
                    "metadata": {
                        "type": "enron_rollout_final",
                        "reward": str(traj.reward),
                        **{k: str(v) for k, v in traj.metrics.items()},
                        **{k: str(v) for k, v in traj.metadata.items()},
                    },
                },
                resp_payload=llm_response,
                status_code=200,
            )
        except Exception as e:
            print(f"Error reporting to OpenPipe: {e}")

    return traj


async def benchmark_model(model: str, limit: int = 10) -> list[Trajectory]:
    import polars as pl
    from query_iterators import load_synthetic_queries

    scenarios = load_synthetic_queries(split="test", limit=limit)
    trajectories = await tqdm.gather(
        *[rollout(model, scenario, trainable=False) for scenario in scenarios],
        desc=f"Benchmarking {model}",
    )

    metrics = pl.DataFrame(
        [t.metrics for t in trajectories],
    )
    # Compute mean for each column (averaging the metrics)
    avg_metrics = metrics.select(
        [pl.mean(c).alias(c) for c in metrics.columns]
    ).transpose(include_header=True)
    print(f"Averaged Metrics for model {model}:")
    print(avg_metrics.to_pandas().to_markdown())
    print(f"Trajectory")
    print(trajectories[0].for_logging())

    return trajectories


if __name__ == "__main__":
    from query_iterators import load_synthetic_queries
    from dotenv import load_dotenv

    load_dotenv()

    import asyncio

    # asyncio.run(
    #     rollout(
    #         "openai/gpt-4o",
    #         load_synthetic_queries(split="test", limit=1)[0],
    #         trainable=False,
    #     )
    # )

    asyncio.run(benchmark_model("openai/gpt-4o"))
    # asyncio.run(benchmark_model("gemini/gemini-2.0-flash"))
    # asyncio.run(benchmark_model("gemini/gemini-2.5-pro-preview-03-25"))
