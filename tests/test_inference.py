import asyncio
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from mlx_serve import inference
from mlx_serve.models import (
    ContentBlockText,
    ContentBlockThinking,
    ContentBlockToolUse,
    Message,
    MessagesRequest,
    ThinkingConfig,
    Tool,
    Usage,
)

_average = inference._average


def test_average_ignores_zero_values():
    assert _average("decode_tps", [{"decode_tps": 0.0}, {"decode_tps": 21.5}]) == 21.5


def test_average_returns_zero_when_no_values_are_measured():
    assert _average("decode_tps", [{"decode_tps": 0.0}]) == 0.0


def make_request(**kwargs):
    values = {
        "model": "mlx-community/test",
        "max_tokens": 8,
        "messages": [Message(role="user", content="hello")],
    }
    values.update(kwargs)
    return MessagesRequest(**values)


def run_async(coroutine):
    return asyncio.run(coroutine)


def test_common_prefix_and_cache_selection_paths(monkeypatch):
    assert inference._common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert inference._common_prefix([1], [2]) == 0

    monkeypatch.setattr(inference.config, "CACHE_MIN_REUSE", 2)
    monkeypatch.setattr(inference.runtime, "prompt_caches", [])
    assert inference._take_cache([1, 2, 3]) == (None, 0)

    cache = [SimpleNamespace(offset=3)]
    inference.runtime.prompt_caches.append({"cache": cache, "tokens": [1, 2]})
    monkeypatch.setattr(
        inference, "can_trim_prompt_cache", lambda value: pytest.fail("no trim needed")
    )
    assert inference._take_cache([1, 2, 3]) == (cache, 2)
    assert inference.runtime.prompt_caches == []


@pytest.mark.parametrize("can_trim", [False, True])
def test_cache_selection_handles_longer_cached_prefix(monkeypatch, can_trim):
    cache = [SimpleNamespace(offset=4)]
    monkeypatch.setattr(inference.config, "CACHE_MIN_REUSE", 2)
    monkeypatch.setattr(
        inference.runtime,
        "prompt_caches",
        [{"cache": cache, "tokens": [1, 2, 3, 4]}],
    )
    trim_calls = []
    monkeypatch.setattr(inference, "can_trim_prompt_cache", lambda value: can_trim)
    monkeypatch.setattr(
        inference, "trim_prompt_cache", lambda value, extra: trim_calls.append(extra)
    )

    selected, common = inference._take_cache([1, 2, 3])

    if can_trim:
        assert selected is cache
        assert common == 2
        assert trim_calls == [2]
    else:
        assert selected is None
        assert common == 0
        assert trim_calls == []


def test_generate_worker_emits_tokens_tracks_stats_and_pools_cache(monkeypatch):
    cache = [SimpleNamespace(offset=3)]
    chunks = [
        SimpleNamespace(
            token=10,
            text="a",
            finish_reason=None,
            generation_tokens=1,
            peak_memory=2.5,
        ),
        SimpleNamespace(
            token=11,
            text="b",
            finish_reason="stop",
            generation_tokens=2,
            peak_memory=2.5,
        ),
    ]
    emitted = []
    monkeypatch.setattr(inference, "_take_cache", lambda ids: (None, 0))
    monkeypatch.setattr(inference, "make_prompt_cache", lambda model: cache)
    monkeypatch.setattr(
        inference, "stream_generate", lambda *args, **kwargs: iter(chunks)
    )
    monkeypatch.setattr(inference.time, "perf_counter", iter([1.0, 2.0, 3.0]).__next__)
    monkeypatch.setattr(inference.runtime, "model", object())
    monkeypatch.setattr(inference.runtime, "tokenizer", object())
    monkeypatch.setattr(inference.runtime, "current_model_name", "mlx-community/test")
    monkeypatch.setattr(inference.runtime, "prompt_caches", [])
    monkeypatch.setattr(inference.runtime, "benchmarks", deque(maxlen=200))
    monkeypatch.setattr(
        inference.runtime,
        "benchmark_totals",
        {"generations": 0, "prompt_tokens": 0, "output_tokens": 0},
    )
    monkeypatch.setattr(inference.config, "CACHE_ENTRIES", 1)

    inference._generate_worker(
        [1, 2], object(), 4, emitted.append, threading.Event(), {}
    )

    assert emitted == [("a", None, 1), ("b", "stop", 2)]
    assert inference.runtime.benchmark_totals == {
        "generations": 1,
        "prompt_tokens": 2,
        "output_tokens": 2,
    }
    assert inference.runtime.benchmarks[0]["peak_memory_gb"] == 2.5
    assert inference.runtime.prompt_caches == [{"cache": cache, "tokens": [1, 2, 10]}]


def test_generate_worker_observes_cancellation_and_empty_generation(monkeypatch):
    cache = [SimpleNamespace()]
    chunks = [
        SimpleNamespace(
            token=10,
            text="first",
            finish_reason=None,
            generation_tokens=1,
            peak_memory=0.0,
        ),
        SimpleNamespace(
            token=11,
            text="second",
            finish_reason=None,
            generation_tokens=2,
            peak_memory=0.0,
        ),
    ]
    emitted = []
    cancel = threading.Event()
    monkeypatch.setattr(inference, "_take_cache", lambda ids: (None, 0))
    monkeypatch.setattr(inference, "make_prompt_cache", lambda model: cache)
    monkeypatch.setattr(
        inference, "stream_generate", lambda *args, **kwargs: iter(chunks)
    )
    monkeypatch.setattr(inference.time, "perf_counter", iter([1.0, 2.0, 3.0]).__next__)
    monkeypatch.setattr(inference.runtime, "model", object())
    monkeypatch.setattr(inference.runtime, "tokenizer", object())
    monkeypatch.setattr(inference.runtime, "current_model_name", "mlx-community/test")
    monkeypatch.setattr(inference.runtime, "prompt_caches", [])
    monkeypatch.setattr(inference.runtime, "benchmarks", deque(maxlen=200))
    monkeypatch.setattr(
        inference.runtime,
        "benchmark_totals",
        {"generations": 0, "prompt_tokens": 0, "output_tokens": 0},
    )

    def emit(item):
        emitted.append(item)
        cancel.set()

    inference._generate_worker([1, 2], None, 4, emit, cancel, {})

    assert emitted == [("first", None, 1)]
    assert inference.runtime.prompt_caches == []

    monkeypatch.setattr(inference, "_take_cache", lambda ids: (cache, 1))
    monkeypatch.setattr(inference, "stream_generate", lambda *args, **kwargs: iter([]))
    monkeypatch.setattr(inference.time, "perf_counter", iter([1.0, 2.0]).__next__)
    stats = {}
    inference._generate_worker(
        [1, 2], None, 4, emitted.append, threading.Event(), stats
    )
    assert stats["output_tokens"] == 0


def test_token_stream_yields_worker_items_and_passes_stats(monkeypatch):
    seen = []

    def worker(prompt_ids, sampler, max_tokens, emit, cancel, stats):
        seen.append((prompt_ids, sampler, max_tokens, stats))
        emit(("text", None, 1))

    monkeypatch.setattr(inference, "_generate_worker", worker)
    monkeypatch.setattr(inference.runtime, "model_epoch", 5)

    async def collect():
        stats = {}
        result = [
            item async for item in inference.token_stream([1], "sampler", 2, 5, stats)
        ]
        return result, stats

    result, stats = run_async(collect())

    assert result == [("text", None, 1)]
    assert seen == [([1], "sampler", 2, stats)]


def test_token_stream_rejects_stale_epoch(monkeypatch):
    monkeypatch.setattr(inference.runtime, "model_epoch", 5)

    async def run():
        with pytest.raises(RuntimeError, match="Model changed"):
            async for _ in inference.token_stream([1], None, 2, 4):
                pass

    run_async(run())


def test_token_stream_propagates_worker_exceptions(monkeypatch):
    def worker(prompt_ids, sampler, max_tokens, emit, cancel, stats):
        emit(ValueError("generation failure"))

    monkeypatch.setattr(inference, "_generate_worker", worker)

    async def run():
        with pytest.raises(ValueError, match="generation failure"):
            async for _ in inference.token_stream([1], None, 2):
                pass

    run_async(run())


def test_token_stream_captures_worker_exception(monkeypatch):
    def worker(*args):
        raise ValueError("worker failed")

    monkeypatch.setattr(inference, "_generate_worker", worker)

    async def run():
        with pytest.raises(ValueError, match="worker failed"):
            async for _ in inference.token_stream([1], None, 2):
                pass

    run_async(run())


def prepare_generation(monkeypatch, chunks, decode="prompt"):
    monkeypatch.setattr(
        inference.runtime,
        "tokenizer",
        SimpleNamespace(decode=lambda ids: decode),
    )
    monkeypatch.setattr(inference.runtime, "current_model_name", "mlx-community/test")
    monkeypatch.setattr(inference.runtime, "benchmarks", [])
    monkeypatch.setattr(inference, "make_sampler", lambda **kwargs: kwargs)

    async def fake_token_stream(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    monkeypatch.setattr(inference, "token_stream", fake_token_stream)


def test_generate_events_yields_thinking_and_parsed_tool_calls(monkeypatch):
    chunks = [
        ("Hi<think>reason</think> there", None, 1),
        ('<tool_call>{"name":"search","arguments":{"q":"x"}}</tool_call>', "stop", 2),
    ]
    prepare_generation(monkeypatch, chunks)
    monkeypatch.setattr(inference.config, "VERBOSE", True)
    request = make_request(
        tools=[Tool(name="search")],
        thinking=ThinkingConfig(type="enabled"),
        temperature=0.4,
        top_p=0.8,
        top_k=2,
    )
    request._request_id = 17

    async def collect():
        return [item async for item in inference.generate_events(request, [1, 2])]

    events = run_async(collect())

    assert [kind for kind, _ in events] == [
        "text",
        "thinking",
        "text",
        "tool_use",
        "done",
    ]
    assert events[3][1]["name"] == "search"
    assert events[4][1]["stop_reason"] == "tool_use"
    assert events[4][1]["output_tokens"] == 2


def test_generate_events_hides_thinking_stops_on_custom_sequence(monkeypatch):
    prepare_generation(
        monkeypatch,
        [("before<think>hidden</think>visibleSTOPignored", None, 3)],
    )
    monkeypatch.setattr(inference.config, "SHOW_THINKING", False)
    request = make_request(stop_sequences=["STOP"])

    async def collect():
        return [item async for item in inference.generate_events(request, [1])]

    events = run_async(collect())

    assert events == [
        ("text", "before"),
        ("text", "visible"),
        (
            "done",
            {
                "stop_reason": "stop_sequence",
                "stop_sequence": "STOP",
                "output_tokens": 3,
            },
        ),
    ]


def test_generate_events_flushes_partial_marker_and_maps_max_tokens(monkeypatch):
    prepare_generation(monkeypatch, [("visible</thi", "length", 7)], decode="<think>")
    monkeypatch.setattr(inference.config, "SHOW_THINKING", True)
    request = make_request()

    async def collect():
        return [item async for item in inference.generate_events(request, [1])]

    events = run_async(collect())

    assert events == [
        ("thinking", "visible"),
        ("thinking", "</thi"),
        (
            "done",
            {
                "stop_reason": "max_tokens",
                "stop_sequence": None,
                "output_tokens": 7,
            },
        ),
    ]


def test_generate_events_surfaces_unparseable_tool_output(monkeypatch, caplog):
    prepare_generation(monkeypatch, [("<tool_call>bad call", None, 1)])
    request = make_request()

    async def collect():
        return [item async for item in inference.generate_events(request, [1])]

    with caplog.at_level("WARNING", logger="proxy"):
        events = run_async(collect())

    assert events[0] == ("text", "<tool_call>bad call")
    assert events[-1][1]["stop_reason"] == "end_turn"
    assert "Unparseable tool call" in caplog.text


def test_generate_response_builds_all_content_blocks(monkeypatch):
    async def events(request, prompt_ids):
        yield "thinking", "reason"
        yield "text", "answer"
        yield "tool_use", {"id": "toolu_1", "name": "search", "input": {"q": "x"}}
        yield (
            "done",
            {
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "output_tokens": 3,
            },
        )

    monkeypatch.setattr(inference, "generate_events", events)

    response = run_async(inference.generate_response(make_request(), [1, 2]))

    assert response.content == [
        ContentBlockThinking(type="thinking", thinking="reason", signature=""),
        ContentBlockText(text="answer"),
        ContentBlockToolUse(id="toolu_1", name="search", input={"q": "x"}),
    ]
    assert response.usage == Usage(input_tokens=2, output_tokens=3)


def test_generate_response_returns_empty_text_when_no_blocks(monkeypatch):
    async def events(request, prompt_ids):
        yield (
            "done",
            {
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "output_tokens": 0,
            },
        )

    monkeypatch.setattr(inference, "generate_events", events)

    response = run_async(inference.generate_response(make_request(), []))

    assert response.content == [ContentBlockText(text="")]


def test_stream_response_transitions_from_thinking_through_tool_to_text(monkeypatch):
    async def events(request, prompt_ids):
        yield "thinking", "reason"
        yield "tool_use", {"id": "toolu_1", "name": "search", "input": {"q": "x"}}
        yield "text", "answer"
        yield (
            "done",
            {
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "output_tokens": 3,
            },
        )

    monkeypatch.setattr(inference, "generate_events", events)

    async def collect():
        return [
            item
            async for item in inference.stream_generate_response(make_request(), [1])
        ]

    output = run_async(collect())
    joined = "".join(output)

    assert "event: message_start" in output[0]
    assert '"type": "signature_delta"' in joined
    assert '"type": "input_json_delta"' in joined
    assert "event: message_delta" in joined
    assert output[-1] == 'event: message_stop\ndata: {"type": "message_stop"}\n\n'


def test_stream_response_closes_text_block_after_tool(monkeypatch):
    async def events(request, prompt_ids):
        yield "text", "answer"
        yield "tool_use", {"id": "toolu_1", "name": "search", "input": {}}
        yield (
            "done",
            {
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "output_tokens": 2,
            },
        )

    monkeypatch.setattr(inference, "generate_events", events)

    async def collect():
        return [
            item
            async for item in inference.stream_generate_response(make_request(), [1])
        ]

    output = run_async(collect())

    assert sum("event: content_block_stop" in item for item in output) == 2


def test_stream_response_emits_error_and_stops(monkeypatch):
    async def events(request, prompt_ids):
        raise RuntimeError("stream failed")
        yield

    monkeypatch.setattr(inference, "generate_events", events)

    async def collect():
        return [
            item
            async for item in inference.stream_generate_response(make_request(), [1])
        ]

    output = run_async(collect())

    assert len(output) == 2
    assert "event: error" in output[1]
    assert "stream failed" in output[1]


def test_stream_response_emits_empty_text_block_when_no_content(monkeypatch):
    async def events(request, prompt_ids):
        yield (
            "done",
            {
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "output_tokens": 0,
            },
        )

    monkeypatch.setattr(inference, "generate_events", events)

    async def collect():
        return [
            item
            async for item in inference.stream_generate_response(make_request(), [])
        ]

    output = run_async(collect())

    assert any(
        '"content_block": {"type": "text", "text": ""}' in item for item in output
    )


def test_stream_response_finishes_thinking_signature(monkeypatch):
    async def events(request, prompt_ids):
        yield "thinking", "reason"
        yield (
            "done",
            {
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "output_tokens": 1,
            },
        )

    monkeypatch.setattr(inference, "generate_events", events)

    async def collect():
        return [
            item
            async for item in inference.stream_generate_response(make_request(), [1])
        ]

    output = run_async(collect())
    assert any('"type": "signature_delta"' in item for item in output)


def test_stream_response_transitions_from_thinking_to_text(monkeypatch):
    async def events(request, prompt_ids):
        yield "thinking", "reason"
        yield "text", "answer"
        yield (
            "done",
            {
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "output_tokens": 2,
            },
        )

    monkeypatch.setattr(inference, "generate_events", events)

    async def collect():
        return [
            item
            async for item in inference.stream_generate_response(make_request(), [1])
        ]

    output = run_async(collect())

    assert sum("event: content_block_stop" in item for item in output) == 2
    assert any('"type": "signature_delta"' in item for item in output)


def test_lifespan_loads_and_shuts_down_runtime(monkeypatch):
    executor = ThreadPoolExecutor(max_workers=1)
    calls = []
    runtime = SimpleNamespace(
        executor=executor,
        load_blocking=lambda name: calls.append(("load", name)),
        shutdown=lambda: shutdown_runtime(executor, calls),
    )
    discovery_finished = threading.Event()
    monkeypatch.setattr(inference, "runtime", runtime)
    monkeypatch.setattr(
        inference, "refresh_discovery", lambda: discovery_finished.set()
    )

    async def run():
        async with inference.lifespan(FastAPI()):
            calls.append(("entered", None))

    run_async(run())

    assert ("load", inference.config.MODEL_NAME) in calls
    assert ("entered", None) in calls
    assert discovery_finished.wait(timeout=1)
    assert ("shutdown", None) in calls


def test_lifespan_shuts_down_after_startup_failure(monkeypatch):
    executor = ThreadPoolExecutor(max_workers=1)
    calls = []
    runtime = SimpleNamespace(
        executor=executor,
        load_blocking=lambda name: (_ for _ in ()).throw(RuntimeError("startup")),
        shutdown=lambda: shutdown_runtime(executor, calls),
    )
    monkeypatch.setattr(inference, "runtime", runtime)
    monkeypatch.setattr(inference, "refresh_discovery", lambda: None)

    async def run():
        with pytest.raises(RuntimeError, match="startup"):
            async with inference.lifespan(FastAPI()):
                pass

    run_async(run())

    assert ("shutdown", None) in calls


async def shutdown_runtime(executor, calls):
    calls.append(("shutdown", None))
    executor.shutdown(wait=False)


def test_lifespan_without_preload_skips_model_load(monkeypatch):
    executor = ThreadPoolExecutor(max_workers=1)
    calls = []
    runtime = SimpleNamespace(
        executor=executor,
        load_blocking=lambda name: calls.append(("load", name)),
        shutdown=lambda: shutdown_runtime(executor, calls),
    )
    monkeypatch.setattr(inference, "runtime", runtime)
    monkeypatch.setattr(inference.config, "PRELOAD", False)
    monkeypatch.setattr(inference, "refresh_discovery", lambda: None)

    async def run():
        async with inference.lifespan(FastAPI()):
            pass

    run_async(run())

    assert ("shutdown", None) in calls
    assert not any(call[0] == "load" for call in calls)
