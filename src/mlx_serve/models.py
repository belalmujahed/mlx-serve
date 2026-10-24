from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, PrivateAttr


class ContentBlockText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"] = "image"
    source: Dict[str, Any]


class ContentBlockThinking(BaseModel):
    type: Literal["thinking", "redacted_thinking"]
    thinking: Optional[str] = None
    signature: Optional[str] = None


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[
        str,
        int,
        float,
        bool,
        Dict[str, Any],
        List[Union[str, int, float, bool, Dict[str, Any], None]],
        None,
    ] = ""


class SystemContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingConfig(BaseModel):
    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: Optional[int] = None


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)


ContentBlock = Union[
    ContentBlockText,
    ContentBlockImage,
    ContentBlockThinking,
    ContentBlockToolUse,
    ContentBlockToolResult,
]


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[ContentBlock]]


class MessagesRequest(BaseModel):
    _model_epoch: int = PrivateAttr(default=0)
    _request_id: int = PrivateAttr(default=0)

    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class MessageResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[Union[ContentBlockThinking, ContentBlockText, ContentBlockToolUse]]
    model: str
    stop_reason: str = "end_turn"
    stop_sequence: Optional[str] = None
    usage: Usage


class ModelSwitchRequest(BaseModel):
    model: str = Field(min_length=1)
