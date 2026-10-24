import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from .catalog import is_model_repo, list_catalog
from .config import config
from .models import (
    Message,
    MessageResponse,
    MessagesRequest,
    ModelSwitchRequest,
    SystemContent,
    ThinkingConfig,
    TokenCountRequest,
    Tool,
)
from .prompts import PromptTokenizer
from .runtime import ModelRuntime

log = logging.getLogger("proxy")


def create_api_router(
    runtime: ModelRuntime,
    format_messages: Callable[
        [
            PromptTokenizer,
            List[Message],
            Optional[Union[str, List[SystemContent]]],
            Optional[List[Tool]],
            Optional[ThinkingConfig],
        ],
        str,
    ],
    encode_prompt: Callable[[PromptTokenizer, str], List[int]],
    generate_response: Callable[
        [MessagesRequest, List[int]], Awaitable[MessageResponse]
    ],
    stream_generate_response: Callable[
        [MessagesRequest, List[int]], AsyncIterator[str]
    ],
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/messages")
    async def create_message(request: MessagesRequest):
        rid = next(runtime.request_ids)
        request._request_id = rid
        log.info(
            "req #%d model=%s stream=%s msgs=%d tools=%d max_tokens=%s thinking=%s",
            rid,
            request.model,
            request.stream,
            len(request.messages),
            len(request.tools or []),
            request.max_tokens,
            request.thinking.type if request.thinking else None,
        )
        try:
            request._model_epoch = await runtime.resolve_model(request.model)
            if runtime.model is None or runtime.tokenizer is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            prompt = format_messages(
                runtime.tokenizer,
                request.messages,
                request.system,
                request.tools,
                request.thinking,
            )
            prompt_ids = encode_prompt(runtime.tokenizer, prompt)

            if request.stream:
                return StreamingResponse(
                    stream_generate_response(request, prompt_ids),
                    media_type="text/event-stream",
                )
            return await generate_response(request, prompt_ids)
        except HTTPException as e:
            log.error("req #%d failed: %s", rid, e.detail)
            raise
        except Exception as e:
            log.exception("req #%d failed", rid)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post("/v1/messages/count_tokens")
    async def count_tokens_endpoint(request: TokenCountRequest):
        if runtime.tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            prompt = format_messages(
                runtime.tokenizer,
                request.messages,
                request.system,
                request.tools,
                request.thinking,
            )
            return {"input_tokens": len(encode_prompt(runtime.tokenizer, prompt))}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    def model_entry(model_id: str, state: Optional[str] = None) -> Dict[str, Any]:
        name = f"{model_id} [{state}]" if state else model_id
        return {
            "type": "model",
            "id": model_id,
            "display_name": name,
            "description": (
                f"Local MLX model ({state})" if state else "Local MLX model"
            ),
            "created_at": "2024-01-01T00:00:00Z",
        }

    @router.get("/v1/models")
    async def list_models(limit: int = Query(default=50, ge=1, le=500)):
        loop = asyncio.get_running_loop()
        catalog = await loop.run_in_executor(
            None, list_catalog, runtime.current_model_name
        )
        data = [model_entry(m["id"], m["state"]) for m in catalog]
        if config.API_MODEL_NAME not in {entry["id"] for entry in data}:
            data.append(model_entry(config.API_MODEL_NAME))
        page = data[:limit]
        return {
            "data": page,
            "has_more": len(data) > len(page),
            "first_id": page[0]["id"],
            "last_id": page[-1]["id"],
        }

    @router.get("/v1/model")
    async def current_model():
        return {
            "model": runtime.current_model_name,
            "loaded": runtime.model is not None,
        }

    @router.post("/v1/model")
    async def switch_model(request: ModelSwitchRequest):
        if not is_model_repo(request.model):
            raise HTTPException(
                status_code=400,
                detail="model must be a Hugging Face repo id like org/name",
            )
        await runtime.switch_model(request.model)
        return {"model": runtime.current_model_name, "loaded": True}

    @router.get("/v1/models/{model_id:path}")
    async def get_model(model_id: str):
        return model_entry(model_id)

    def average(key: str, rows: List[Dict[str, Any]]) -> float:
        values = [row[key] for row in rows if row[key]]
        return round(sum(values) / len(values), 2) if values else 0.0

    @router.get("/stats")
    async def stats():
        """Benchmarks for the loaded model over the most recent generations."""
        rows = [
            row
            for row in runtime.benchmarks
            if row["model"] == runtime.current_model_name
        ]
        return {
            "model": runtime.current_model_name,
            "lifetime": runtime.benchmark_totals,
            "recent_generations": len(rows),
            "avg_ttft_s": average("ttft_s", rows),
            "avg_prefill_tps": average("prefill_tps", rows),
            "avg_decode_tps": average("decode_tps", rows),
            "max_peak_memory_gb": max(
                (row["peak_memory_gb"] for row in rows), default=0.0
            ),
            "last": list(runtime.benchmarks)[-10:],
        }

    @router.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "model_loaded": runtime.model is not None,
            "model": runtime.current_model_name,
        }

    @router.get("/")
    async def root():
        return {
            "message": "Claude Code MLX Proxy",
            "status": "running",
            "model_loaded": runtime.model is not None,
        }

    return router
