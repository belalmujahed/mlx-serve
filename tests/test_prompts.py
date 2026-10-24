import pytest
from fastapi import HTTPException

from mlx_serve.models import (
    ContentBlockImage,
    ContentBlockText,
    ContentBlockThinking,
    ContentBlockToolResult,
    ContentBlockToolUse,
    Message,
    SystemContent,
    ThinkingConfig,
    Tool,
)
from mlx_serve.prompts import (
    _tool_result_text,
    convert_messages,
    encode_prompt,
    extract_system_text,
    format_messages,
    format_tools,
)


class FakeTokenizer:
    chat_template = None
    bos_token = None

    def apply_chat_template(self, conversation, **kwargs):
        raise AssertionError("fallback formatting should be used")

    def encode(self, prompt, add_special_tokens):
        return [len(prompt), int(add_special_tokens)]


def test_convert_messages_preserves_system_text_and_tool_results():
    messages = [
        Message(
            role="assistant",
            content=[
                ContentBlockText(text="Searching"),
            ],
        ),
        Message(
            role="user",
            content=[
                ContentBlockToolResult(tool_use_id="toolu_1", content="Found it"),
                ContentBlockText(text="Thanks"),
            ],
        ),
    ]

    assert convert_messages(messages, [SystemContent(text="Be concise")]) == [
        {"role": "system", "content": "Be concise"},
        {"role": "assistant", "content": "Searching"},
        {"role": "tool", "content": "Found it"},
        {"role": "user", "content": "Thanks"},
    ]


def test_format_messages_fallback_renders_system_and_user():
    prompt = format_messages(
        FakeTokenizer(),
        [Message(role="user", content="Hello")],
        system="Be concise",
    )

    assert prompt == (
        "<|system|>\nBe concise\n<|end|>\n<|user|>\nHello\n<|end|>\n<|assistant|>\n"
    )


def test_encode_prompt_uses_tokenizer_bos_setting():
    tokenizer = FakeTokenizer()

    assert encode_prompt(tokenizer, "Hello") == [5, 1]
    tokenizer.bos_token = "<s>"
    assert encode_prompt(tokenizer, "<s>Hello") == [8, 0]


def test_tool_input_schema_default_is_not_shared():
    first = Tool(name="first")
    second = Tool(name="second")

    first.input_schema["type"] = "object"

    assert second.input_schema == {}


def test_extract_system_text_handles_string_list_and_missing_value():
    assert extract_system_text("system prompt") == "system prompt"
    assert (
        extract_system_text([SystemContent(text="first"), SystemContent(text="second")])
        == "first\n\nsecond"
    )
    assert extract_system_text(None) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("plain", "plain"),
        ({"type": "text", "text": "hello"}, "hello"),
        ({"type": "image"}, "[image]"),
        (
            [{"type": "text", "text": "hello"}, "world", {"type": "image"}],
            "hello\nworld\n[image]",
        ),
        (None, ""),
        (42, "42"),
    ],
)
def test_tool_result_text_converts_supported_content(value, expected):
    assert _tool_result_text(value) == expected


def test_convert_messages_handles_tool_use_images_and_ignored_blocks():
    messages = [
        Message(
            role="assistant",
            content=[
                ContentBlockToolUse(
                    id="toolu_1", name="search", input={"query": "hello"}
                ),
                ContentBlockThinking(type="thinking", thinking="reasoning"),
            ],
        ),
        Message(
            role="user",
            content=[
                ContentBlockImage(source={"type": "base64", "data": "x"}),
            ],
        ),
        Message(role="assistant", content=[ContentBlockText(text="")]),
    ]

    assert convert_messages(messages, None) == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": {"query": "hello"},
                    },
                }
            ],
        },
        {"role": "user", "content": "[image]"},
    ]


def test_format_tools_returns_none_or_function_descriptions():
    assert format_tools(None) is None
    assert format_tools([]) is None
    assert format_tools(
        [
            Tool(
                name="search",
                description="Find things",
                input_schema={"type": "object"},
            )
        ]
    ) == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Find things",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert format_tools([Tool(name="empty")])[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {},
    }


def test_format_messages_uses_chat_template_when_it_returns_text():
    class ChatTokenizer(FakeTokenizer):
        chat_template = "template"

        def apply_chat_template(self, conversation, **kwargs):
            self.args = (conversation, kwargs)
            return "templated prompt"

    tokenizer = ChatTokenizer()
    prompt = format_messages(
        tokenizer,
        [Message(role="user", content="hello")],
        tools=[Tool(name="search")],
        thinking=ThinkingConfig(type="enabled"),
    )

    assert prompt == "templated prompt"
    assert tokenizer.args[1]["enable_thinking"] is True
    assert tokenizer.args[1]["tools"][0]["function"]["name"] == "search"


def test_format_messages_falls_back_when_template_returns_token_ids():
    class TokenIdTokenizer(FakeTokenizer):
        chat_template = "template"

        def apply_chat_template(self, conversation, **kwargs):
            return [1, 2, 3]

    prompt = format_messages(
        TokenIdTokenizer(),
        [Message(role="user", content="hello")],
        thinking=ThinkingConfig(type="disabled"),
    )

    assert prompt.endswith("<|assistant|>\n")


def test_format_messages_logs_template_error_and_falls_back(caplog):
    class BrokenTokenizer(FakeTokenizer):
        chat_template = "template"

        def apply_chat_template(self, conversation, **kwargs):
            raise ValueError("broken template")

    with caplog.at_level("WARNING", logger="proxy"):
        prompt = format_messages(
            BrokenTokenizer(), [Message(role="user", content="hello")]
        )

    assert "Chat template failed (broken template)" in caplog.text
    assert "<|user|>\nhello" in prompt


def test_format_messages_does_not_drop_tools_after_template_failure():
    class BrokenTokenizer(FakeTokenizer):
        chat_template = "template"

        def apply_chat_template(self, conversation, **kwargs):
            raise ValueError("broken template")

    with pytest.raises(HTTPException, match="Chat template failed with tools"):
        format_messages(
            BrokenTokenizer(),
            [Message(role="user", content="hello")],
            tools=[Tool(name="search")],
        )
