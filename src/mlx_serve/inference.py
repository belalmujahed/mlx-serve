import asyncio
import json
import logging
import threading
import time
from collections.abc import Callable
from contextlib import aclosing, asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI
from mlx_lm import stream_generate
from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.sample_utils import make_sampler

from .api import create_api_router
from .catalog import refresh_discovery
from .config import config
from .models import (
    ContentBlockText,
    ContentBlockThinking,
    ContentBlockToolUse,
    MessageResponse,
    MessagesRequest,
    Usage,
)
from .prompts import encode_prompt, format_messages
from .runtime import runtime
from .tool_calls import StreamParser, new_id, parse_tool_calls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("proxy")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, refresh_discovery)
        if config.PRELOAD:
            await loop.run_in_executor(
                runtime.executor, runtime.load_blocking, config.MODEL_NAME
            )
        else:
            log.info("Started without a model; one loads on the first request")
        yield
    finally:
        await runtime.shutdown()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_DONE = object()


def _average(key: str, rows: List[Dict[str, Any]]) -> float:
    values = [row[key] for row in rows if row[key]]
    return round(sum(values) / len(values), 2) if values else 0.0


def _common_prefix(a: List[int], b: List[int]) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def _take_cache(prompt_ids: List[int]) -> Tuple[Optional[Any], int]:
    """Remove and return the pooled cache sharing the longest useful prefix."""
    best, best_common = None, 0
    for entry in runtime.prompt_caches:
        common = _common_prefix(entry["tokens"], prompt_ids)
        if common > best_common:
            best, best_common = entry, common
    if best is None or best_common < config.CACHE_MIN_REUSE:
        return None, 0

    runtime.prompt_caches.remove(best)
    # At least one token must be processed to produce logits.
    common = min(best_common, len(prompt_ids) - 1)
    extra = len(best["tokens"]) - common
    if extra and not can_trim_prompt_cache(best["cache"]):
        return None, 0
    if extra:
        trim_prompt_cache(best["cache"], extra)
    return best["cache"], common


def _generate_worker(
    prompt_ids: List[int],
    sampler: Any,
    max_tokens: int,
    emit: Callable[[Any], None],
    cancel: threading.Event,
    stats: Dict[str, Any],
) -> None:
    """Runs on the MLX worker thread. Reuses a pooled KV cache for the shared prefix."""
    start = time.perf_counter()
    cache, common = _take_cache(prompt_ids)
    if cache is None:
        cache, common = make_prompt_cache(runtime.model), 0

    generated: List[int] = []
    first_token_at = None
    last = None
    for response in stream_generate(
        runtime.model,
        runtime.tokenizer,
        prompt=prompt_ids[common:],
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=cache,
    ):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        generated.append(response.token)
        last = response
        emit((response.text, response.finish_reason, response.generation_tokens))
        if cancel.is_set():
            break

    end = time.perf_counter()
    ttft = (first_token_at or end) - start
    decode_time = end - (first_token_at or end)
    # Decode speed excludes the first token, which is produced by prefill.
    decode_tps = (len(generated) - 1) / decode_time if decode_time > 0 else 0.0
    prefill_tps = (
        (len(prompt_ids) - common) / ttft
        if ttft > 0 and len(prompt_ids) > common
        else 0.0
    )
    peak_gb = last.peak_memory if last is not None else 0.0
    log.info(
        "gen prompt=%d reused=%d (%.0f%%) ttft=%.2fs prefill=%.0f tok/s "
        "out=%d decode=%.1f tok/s total=%.1fs peak_mem=%.1fGB",
        len(prompt_ids),
        common,
        100 * common / max(len(prompt_ids), 1),
        ttft,
        prefill_tps,
        len(generated),
        decode_tps,
        end - start,
        peak_gb,
    )
    entry = {
        "model": runtime.current_model_name,
        "prompt_tokens": len(prompt_ids),
        "cached_tokens": common,
        "output_tokens": len(generated),
        "ttft_s": round(ttft, 3),
        "prefill_tps": round(prefill_tps, 1),
        "decode_tps": round(decode_tps, 1),
        "total_s": round(end - start, 3),
        "peak_memory_gb": round(peak_gb, 2),
    }
    stats.update(entry)
    runtime.benchmarks.append(entry)
    runtime.benchmark_totals["generations"] += 1
    runtime.benchmark_totals["prompt_tokens"] += len(prompt_ids)
    runtime.benchmark_totals["output_tokens"] += len(generated)

    offset = getattr(cache[0], "offset", None)
    if offset is not None:
        runtime.prompt_caches.append(
            {"cache": cache, "tokens": (list(prompt_ids) + generated)[:offset]}
        )
        del runtime.prompt_caches[: -config.CACHE_ENTRIES]


async def token_stream(
    prompt_ids: List[int],
    sampler: Any,
    max_tokens: int,
    epoch: Optional[int] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[Tuple[str, Optional[str], int]]:
    """Async iterator over (text, finish_reason, generated_token_count)."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()

    def emit(item: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def worker() -> None:
        try:
            _generate_worker(
                prompt_ids,
                sampler,
                max_tokens,
                emit,
                cancel,
                stats if stats is not None else {},
            )
        except Exception as e:
            emit(e)
        finally:
            emit(_DONE)

    async with runtime.generation_lock:
        if epoch is not None and epoch != runtime.model_epoch:
            raise RuntimeError("Model changed while the request was queued; retry")
        future = loop.run_in_executor(runtime.executor, worker)
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            cancel.set()
            await asyncio.shield(future)


async def generate_events(
    request: MessagesRequest, prompt_ids: List[int]
) -> AsyncIterator[Tuple[str, Any]]:
    """Yield ("text", str), ("tool_use", dict) and a final ("done", dict)."""
    tools = {t.name: t.input_schema for t in (request.tools or [])}
    # Chat templates for reasoning models may already open the <think> block.
    prompt_tail = runtime.tokenizer.decode(prompt_ids[-6:]).rstrip()
    parser = StreamParser(
        request.stop_sequences, in_thinking=prompt_tail.endswith("<think>")
    )
    if request.thinking is not None:
        show_thinking = request.thinking.type != "disabled"
    else:
        show_thinking = config.SHOW_THINKING

    def visible(pieces):
        return [
            (kind, text)
            for kind, text in pieces
            if text and (kind == "text" or show_thinking)
        ]

    sampler = make_sampler(
        temp=(
            request.temperature
            if request.temperature is not None
            else config.DEFAULT_TEMPERATURE
        ),
        top_p=request.top_p if request.top_p is not None else config.DEFAULT_TOP_P,
        top_k=request.top_k or 0,
    )

    finish_reason: Optional[str] = None
    output_tokens = 0
    raw_output: List[str] = []
    gen_stats: Dict[str, Any] = {}
    request_start = time.perf_counter()
    async with aclosing(
        token_stream(
            prompt_ids, sampler, request.max_tokens, request._model_epoch, gen_stats
        )
    ) as stream:
        async for text, reason, count in stream:
            output_tokens = count
            finish_reason = reason or finish_reason
            raw_output.append(text)
            for kind, piece in visible(parser.feed(text)):
                yield kind, piece
            if parser.stop_hit:
                break

    for kind, piece in visible(parser.finish()):
        yield kind, piece

    calls = parse_tool_calls(parser.tool_buf, tools) if parser.in_tool else []
    recent = [r for r in runtime.benchmarks if r["model"] == runtime.current_model_name]
    log.info(
        "req #%d done in %.1fs: %d in / %d out, ttft=%.2fs, decode=%.1f tok/s "
        "(avg decode=%.1f tok/s, avg ttft=%.2fs over %d runs)",
        request._request_id,
        time.perf_counter() - request_start,
        gen_stats.get("prompt_tokens", 0),
        gen_stats.get("output_tokens", 0),
        gen_stats.get("ttft_s", 0),
        gen_stats.get("decode_tps", 0),
        _average("decode_tps", recent),
        _average("ttft_s", recent),
        len(recent),
    )
    log.info(
        "req #%d output tools_offered=%d tool_calls=%s finish=%s thinking=%s",
        request._request_id,
        len(tools),
        [c["name"] for c in calls] or 0,
        finish_reason,
        show_thinking,
    )
    if parser.in_tool and not calls:
        log.warning("Unparseable tool call: %r", parser.tool_buf[:500])
    if config.VERBOSE:
        log.info("raw output:\n%s", "".join(raw_output))
    if parser.in_tool and not calls:
        # Unparseable tool call: surface the raw text rather than dropping it.
        yield "text", parser.tool_buf
    for call in calls:
        yield "tool_use", {"id": new_id("toolu"), **call}

    if calls:
        stop_reason = "tool_use"
    elif parser.stop_hit:
        stop_reason = "stop_sequence"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    yield (
        "done",
        {
            "stop_reason": stop_reason,
            "stop_sequence": parser.stop_hit,
            "output_tokens": output_tokens,
        },
    )


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def generate_response(
    request: MessagesRequest, prompt_ids: List[int]
) -> MessageResponse:
    """Generate non-streaming response"""
    text_parts: List[str] = []
    thinking_parts: List[str] = []
    blocks: List[
        Union[ContentBlockThinking, ContentBlockText, ContentBlockToolUse]
    ] = []
    done: Dict[str, Any] = {}

    async for kind, data in generate_events(request, prompt_ids):
        if kind == "text":
            text_parts.append(data)
        elif kind == "thinking":
            thinking_parts.append(data)
        elif kind == "tool_use":
            blocks.append(ContentBlockToolUse(**data))
        else:
            done = data

    text = "".join(text_parts)
    if text or not blocks:
        blocks.insert(0, ContentBlockText(text=text))
    if thinking_parts:
        blocks.insert(
            0,
            ContentBlockThinking(
                type="thinking", thinking="".join(thinking_parts), signature=""
            ),
        )

    return MessageResponse(
        id=new_id("msg"),
        content=blocks,
        model=request.model,
        stop_reason=done["stop_reason"],
        stop_sequence=done["stop_sequence"],
        usage=Usage(input_tokens=len(prompt_ids), output_tokens=done["output_tokens"]),
    )


async def stream_generate_response(
    request: MessagesRequest, prompt_ids: List[int]
) -> AsyncIterator[str]:
    """Generate streaming response"""
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": new_id("msg"),
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": request.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": len(prompt_ids), "output_tokens": 0},
            },
        },
    )

    index = -1
    open_kind: Optional[str] = None
    done: Dict[str, Any] = {}

    def block_stop(i: int) -> str:
        return _sse("content_block_stop", {"type": "content_block_stop", "index": i})

    try:
        async with aclosing(generate_events(request, prompt_ids)) as events:
            async for kind, data in events:
                if kind in ("text", "thinking"):
                    if open_kind != kind:
                        if open_kind == "thinking":
                            yield _sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": index,
                                    "delta": {
                                        "type": "signature_delta",
                                        "signature": "",
                                    },
                                },
                            )
                        if open_kind:
                            yield block_stop(index)
                        index += 1
                        open_kind = kind
                        yield _sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": index,
                                "content_block": (
                                    {"type": "thinking", "thinking": ""}
                                    if kind == "thinking"
                                    else {"type": "text", "text": ""}
                                ),
                            },
                        )
                    delta = (
                        {"type": "thinking_delta", "thinking": data}
                        if kind == "thinking"
                        else {"type": "text_delta", "text": data}
                    )
                    yield _sse(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": index, "delta": delta},
                    )
                elif kind == "tool_use":
                    if open_kind == "thinking":
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {"type": "signature_delta", "signature": ""},
                            },
                        )
                    if open_kind:
                        yield block_stop(index)
                        open_kind = None
                    index += 1
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {
                                "type": "tool_use",
                                "id": data["id"],
                                "name": data["name"],
                                "input": {},
                            },
                        },
                    )
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(data["input"]),
                            },
                        },
                    )
                    yield block_stop(index)
                else:
                    done = data
    except Exception as e:
        yield _sse(
            "error",
            {"type": "error", "error": {"type": "api_error", "message": str(e)}},
        )
        return

    if open_kind == "thinking":
        yield _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "signature_delta", "signature": ""},
            },
        )
    if open_kind:
        yield block_stop(index)
    elif index < 0:
        yield _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        yield block_stop(0)

    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": done["stop_reason"],
                "stop_sequence": done["stop_sequence"],
            },
            "usage": {"output_tokens": done["output_tokens"]},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


app.include_router(
    create_api_router(
        runtime,
        format_messages,
        encode_prompt,
        generate_response,
        stream_generate_response,
    )
)
