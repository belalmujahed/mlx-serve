from mlx_serve.tool_calls import (
    StreamParser,
    _coerce,
    _parse_one,
    _schema_props,
    new_id,
    parse_tool_calls,
)


def test_stream_parser_handles_markers_split_across_chunks():
    parser = StreamParser(stop_sequences=["END"])

    assert parser.feed("hello<thi") == [("text", "hello")]
    assert parser.feed("nk>reason</think>answerE") == [
        ("thinking", "reason"),
        ("text", "answer"),
    ]
    assert parser.feed("NDignored") == []
    assert parser.stop_hit == "END"


def test_stream_parser_captures_tool_call():
    parser = StreamParser()

    assert parser.feed('before<tool_call>{"name":"search","arguments":{"q":"x"}}') == [
        ("text", "before")
    ]
    assert parser.in_tool
    assert parse_tool_calls(parser.tool_buf, {}) == [
        {"name": "search", "input": {"q": "x"}}
    ]


def test_parse_json_tool_call_accepts_stringified_arguments():
    assert parse_tool_calls(
        '<tool_call>{"name":"search","arguments":"{\\"limit\\":3}"}</tool_call>',
        {},
    ) == [{"name": "search", "input": {"limit": 3}}]


def test_parse_function_tool_call_coerces_schema_values():
    tools = {"lookup": {"properties": {"query": {"type": "string"}}}}
    output = (
        "<tool_call><function=lookup>"
        "<parameter=query>123</parameter>"
        "<parameter=limit>4</parameter>"
        "</function></tool_call>"
    )

    assert parse_tool_calls(output, tools) == [
        {"name": "lookup", "input": {"query": "123", "limit": 4}}
    ]


def test_parse_glm_tool_call():
    output = (
        "<tool_call>lookup"
        "<arg_key>enabled</arg_key><arg_value>true</arg_value>"
        "</tool_call>"
    )

    assert parse_tool_calls(output, {}) == [
        {"name": "lookup", "input": {"enabled": True}}
    ]


def test_parse_skips_invalid_and_empty_tool_calls():
    assert (
        parse_tool_calls("<tool_call></tool_call><tool_call>{invalid}</tool_call>", {})
        == []
    )


def test_new_id_uses_prefix_and_unique_hex_suffix():
    first = new_id("toolu")
    second = new_id("toolu")

    assert first.startswith("toolu_")
    assert len(first) == len("toolu_") + 24
    assert first != second


def test_stream_parser_handles_thinking_transition_and_flush():
    parser = StreamParser(in_thinking=True)

    assert parser.feed("reason</think>\nanswer<think>more") == [
        ("thinking", "reason"),
        ("text", "answer"),
        ("thinking", "more"),
    ]
    assert parser.in_thinking
    assert parser.finish() == []


def test_stream_parser_holds_partial_marker_and_flushes_on_finish():
    parser = StreamParser()

    assert parser.feed("plain<tool_") == [("text", "plain")]
    assert parser.pending == "<tool_"
    assert parser.finish() == [("text", "<tool_")]


def test_stream_parser_tool_mode_buffers_until_end():
    parser = StreamParser()
    parser.feed("<tool_call>partial")

    assert parser.feed(" continuation") == []
    assert parser.tool_buf == "<tool_call>partial continuation"
    assert parser.finish() == []


def test_stream_parser_stop_sequence_takes_precedence_and_suppresses_finish():
    parser = StreamParser(stop_sequences=["STOP"])

    assert parser.feed("beforeSTOPafter") == [("text", "before")]
    assert parser.stop_hit == "STOP"
    assert parser.finish() == []


def test_tool_call_json_accepts_fallback_argument_keys_and_invalid_string():
    assert parse_tool_calls(
        '<tool_call>{"name":"one","input":{"a":1}}</tool_call>'
        '<tool_call>{"name":"two","parameters":"bad"}</tool_call>',
        {},
    ) == [
        {"name": "one", "input": {"a": 1}},
        {"name": "two", "input": {}},
    ]


def test_tool_call_json_rejects_missing_name_or_non_mapping_arguments():
    assert _parse_one('{"arguments":{"a":1}}', {}) is None
    assert _parse_one('{"name":"tool","arguments":[1]}', {}) is None
    assert _parse_one("not json", {}) is None
    assert _parse_one("  ", {}) is None


def test_tool_call_function_and_glm_variants_cover_schema_and_empty_args():
    tools = {"known": {"properties": {}}}

    assert _parse_one("<function=run><parameter=x>not-json</parameter>", {}) == {
        "name": "run",
        "input": {"x": "not-json"},
    }
    assert _parse_one("known", tools) == {"name": "known", "input": {}}
    assert _parse_one("unknown", {}) is None
    assert _parse_one("bad name", {}) is None
    assert _schema_props({"known": {}}, "known") == {}


def test_coerce_preserves_string_schema_and_handles_json_errors():
    assert _coerce("123", {"type": "string"}) == "123"
    assert _coerce("true", None) is True
    assert _coerce("not-json", None) == "not-json"
    assert _coerce(None, None) is None
