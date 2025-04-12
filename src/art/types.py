# Remove the direct import of Choice
# from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
import pydantic
# Import Protocol and runtime_checkable
from typing import Iterable, Literal, Coroutine, Any, Protocol, runtime_checkable
import asyncio

# --- Define Protocols for Structural Typing ---

@runtime_checkable
class FunctionCallProtocol(Protocol):
    name: str
    arguments: str

@runtime_checkable
class ToolCallProtocol(Protocol):
    id: str
    function: FunctionCallProtocol
    # type: Literal["function"] # Optional: Add if needed for stricter checking

@runtime_checkable
class MessageProtocol(Protocol):
    role: str
    content: str | None
    tool_calls: list[ToolCallProtocol] | None = None

@runtime_checkable
class ChoiceProtocol(Protocol):
    """A protocol representing the structure of a Choice object."""
    message: MessageProtocol
    # Add other attributes if they are used elsewhere (e.g., finish_reason, index)
    # finish_reason: str | None = None
    # index: int = 0


# --- Existing Code ---

BaseModel = Literal[
    # ... (rest of the models)
]

Message = ChatCompletionMessageParam
# Update MessageOrChoice to use the protocol
MessageOrChoice = Message | ChoiceProtocol
Messages = list[Message]
MessagesAndChoices = list[MessageOrChoice]
ToolCall = ChatCompletionMessageToolCallParam
Tools = Iterable[ChatCompletionToolParam]


class Trajectory(pydantic.BaseModel):
    messages_and_choices: MessagesAndChoices
    reward: float
    metrics: dict[str, float] = {}
    tools: Tools | None = None

    def __str__(self) -> str:
        # ... (no changes needed here)

    def pretty_str(self) -> str:
        printed = self.__str__()

        for i, item in enumerate(self.messages_and_choices):
            # Check if it conforms to ChoiceProtocol using isinstance
            # or rely on static analysis and attribute access
            is_choice = isinstance(item, ChoiceProtocol)

            printed += "\n--------"
            if is_choice:
                # Access attributes via the protocol
                message = item.message
                printed += f"Choice {i} ({message.role})\n"
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        printed += f"  Tool call {tool_call.id}: {tool_call.function.name}\n"
                        printed += f"    Args: {tool_call.function.arguments}\n"
                if message.content:
                    printed += f"  Content: {message.content}\n"
            else:
                # Handle the original Message type (which is a dict)
                message_dict = dict(item) # Ensure it's a dict for consistent access
                printed += (
                    f"Message {i} ({message_dict['role']}) trainable: True\n"
                )
                if tool_calls := message_dict.get("tool_calls"):
                    for tool_call in tool_calls:
                        # Access as dict keys
                        printed += f"  Tool call {tool_call['id']}: {tool_call['function']['name']}\n" # type: ignore
                        printed += f"    Args: {tool_call['function']['arguments']}\n" # type: ignore
                if content := message_dict.get("content"):
                     printed += f"  Content: {content}\n"


        return printed


class TrajectoryGroup(pydantic.BaseModel):
    # ... (no changes needed here)


class TuneConfig(pydantic.BaseModel):
    # ... (no changes needed here)


Verbosity = Literal[0, 1, 2] 