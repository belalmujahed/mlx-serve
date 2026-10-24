import logging
from typing import Any, Dict, List, Optional, Protocol, Union

from fastapi import HTTPException

from .models import (
    ContentBlockImage,
    ContentBlockText,
    ContentBlockToolResult,
    ContentBlockToolUse,
    Message,
    SystemContent,
    ThinkingConfig,
    Tool,
)

log = logging.getLogger("proxy")


class PromptTokenizer(Protocol):
    @property
    def chat_template(self) -> object | None: ...

    @property
    def bos_token(self) -> str | None: ...

    def apply_chat_template(
        self, conversation: List[Dict[str, Any]], **kwargs: Any
    ) -> str | List[int]: ...

    def encode(self, prompt: str, add_special_tokens: bool = True) -> List[int]: ...


def extract_system_text(
    system: Optional[Union[str, List[SystemContent]]],
) -> Optional[str]:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n\n".join(content.text for content in system)
    return None


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image":
                    parts.append("[image]")
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if content is None else str(content)


def convert_messages(
    messages: List[Message], system: Optional[Union[str, List[SystemContent]]]
) -> List[Dict[str, Any]]:
    """Convert Anthropic-style messages to chat-template messages."""
    converted: List[Dict[str, Any]] = []

    system_text = extract_system_text(system)
    if system_text:
        converted.append({"role": "system", "content": system_text})

    for message in messages:
        if isinstance(message.content, str):
            converted.append({"role": message.role, "content": message.content})
            continue

        texts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        tool_results: List[Dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, ContentBlockText):
                texts.append(block.text)
            elif isinstance(block, ContentBlockToolUse):
                tool_calls.append(
                    {
                        "type": "function",
                        "function": {"name": block.name, "arguments": block.input},
                    }
                )
            elif isinstance(block, ContentBlockToolResult):
                tool_results.append(
                    {"role": "tool", "content": _tool_result_text(block.content)}
                )
            elif isinstance(block, ContentBlockImage):
                texts.append("[image]")

        text = "\n".join(t for t in texts if t)
        if message.role == "assistant":
            entry: Dict[str, Any] = {"role": "assistant", "content": text}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            if text or tool_calls:
                converted.append(entry)
        else:
            converted.extend(tool_results)
            if text or not tool_results:
                converted.append({"role": message.role, "content": text})

    return converted


def format_tools(tools: Optional[List[Tool]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
            },
        }
        for tool in tools
    ]


def format_messages(
    tokenizer: PromptTokenizer,
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]] = None,
    tools: Optional[List[Tool]] = None,
    thinking: Optional[ThinkingConfig] = None,
) -> str:
    """Render messages (and tool definitions) with the model's chat template."""
    formatted = convert_messages(messages, system)
    template_kwargs: Dict[str, Any] = {}
    if thinking is not None:
        template_kwargs["enable_thinking"] = thinking.type != "disabled"

    if tokenizer.chat_template is not None:
        try:
            result = tokenizer.apply_chat_template(
                formatted,
                tools=format_tools(tools),
                add_generation_prompt=True,
                tokenize=False,
                **template_kwargs,
            )
            if isinstance(result, str):
                return result
        except Exception as e:
            if tools:
                raise HTTPException(
                    status_code=500,
                    detail=f"Chat template failed with tools ({e}); refusing to "
                    "drop tool definitions silently",
                ) from e
            log.warning("Chat template failed (%s); using fallback formatting", e)

    prompt = ""
    for msg in formatted:
        role = msg["role"] if msg["role"] in ("system", "user", "assistant") else "user"
        prompt += f"<|{role}|>\n{msg['content']}\n<|end|>\n"
    prompt += "<|assistant|>\n"
    return prompt


def encode_prompt(tokenizer: PromptTokenizer, prompt: str) -> List[int]:
    add_special = tokenizer.bos_token is None or not prompt.startswith(
        tokenizer.bos_token
    )
    return list(tokenizer.encode(prompt, add_special_tokens=add_special))
