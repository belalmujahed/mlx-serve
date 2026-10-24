import asyncio
import gc
import itertools
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import mlx.core as mx
from fastapi import HTTPException
from huggingface_hub import snapshot_download
from mlx_lm import load, stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS, load_model
from transformers import PreTrainedTokenizerFast

from .catalog import is_model_repo
from .config import config

log = logging.getLogger("proxy")


def get_model_path(name: str) -> Path:
    """Resolve a local directory or download the Hugging Face repository."""
    path = Path(name)
    if path.exists():
        return path
    return Path(snapshot_download(name, allow_patterns=DEFAULT_ALLOW_PATTERNS))


class ModelRuntime:
    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.current_model_name: Optional[str] = None
        self.load_failure_reason: Optional[str] = None
        self.model_epoch = 0
        self.switch_lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self.generation_lock = asyncio.Lock()
        self.prompt_caches: list[dict[str, Any]] = []
        self.request_ids = itertools.count(1)
        self.benchmarks: Deque[Dict[str, Any]] = deque(maxlen=200)
        self.benchmark_totals = {
            "generations": 0,
            "prompt_tokens": 0,
            "output_tokens": 0,
        }

    def _load_model_with_fallback(
        self, model_name: str, tokenizer_config: dict[str, Any]
    ) -> tuple[Any, Any]:
        """Load a tokenizer with the fast-tokenizer fallback if required."""
        try:
            return load(model_name, tokenizer_config=tokenizer_config)
        except ValueError as e:
            if "TokenizersBackend" not in str(e):
                raise
            log.warning(
                "AutoTokenizer cannot load this tokenizer; retrying with "
                "PreTrainedTokenizerFast"
            )

        model_path = get_model_path(model_name)
        mlx_model, mlx_config = load_model(model_path)
        hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(model_path), **tokenizer_config
        )
        tokenizer = TokenizerWrapper(
            hf_tokenizer, eos_token_ids=mlx_config.get("eos_token_id")
        )
        return mlx_model, tokenizer

    def load_blocking(self, name: str) -> None:
        """Load and warm up a model on the MLX worker thread."""
        tokenizer_config = {}
        if config.TRUST_REMOTE_CODE:
            tokenizer_config["trust_remote_code"] = True
        if config.EOS_TOKEN:
            tokenizer_config["eos_token"] = config.EOS_TOKEN

        log.info("Loading model %s (downloads first if not cached)", name)
        started = time.perf_counter()
        try:
            self.model, self.tokenizer = self._load_model_with_fallback(
                name, tokenizer_config
            )
            self.current_model_name = name
        except Exception as e:
            self.load_failure_reason = str(e)
            raise
        log.info("Model loaded in %.1fs; warming up", time.perf_counter() - started)
        started = time.perf_counter()
        self.warmup()
        log.info("Warmup done in %.1fs", time.perf_counter() - started)

    def unload_blocking(self) -> None:
        """Drop the current model, tokenizer, and KV caches on the MLX thread."""
        self.prompt_caches.clear()
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        gc.collect()
        mx.clear_cache()

    def warmup(self) -> None:
        """Page weights into memory and compile kernels before first request."""
        mx.eval(self.model.parameters())
        vocab = min(getattr(self.tokenizer, "vocab_size", 32000), 32000)
        prompt = [(i * 7919) % vocab for i in range(1024)]
        for _ in stream_generate(
            self.model, self.tokenizer, prompt=prompt, max_tokens=4
        ):
            pass

    async def switch_model(self, name: str) -> None:
        """Replace the active model, restoring the previous model on failure."""
        loop = asyncio.get_running_loop()
        async with self.switch_lock:
            if name == self.current_model_name:
                return
            previous = self.current_model_name
            async with self.generation_lock:
                log.info("Switching model: %s -> %s", previous, name)
                await loop.run_in_executor(self.executor, self.unload_blocking)
                self.model_epoch += 1
                try:
                    await loop.run_in_executor(self.executor, self.load_blocking, name)
                except Exception as e:
                    log.error("Failed to load %s: %s", name, e)
                    if previous:
                        await loop.run_in_executor(
                            self.executor, self.load_blocking, previous
                        )
                    raise HTTPException(
                        status_code=502, detail=f"Could not load model {name}: {e}"
                    ) from e

    async def resolve_model(self, requested: str) -> int:
        """Switch for Hugging Face repo IDs and return the active model epoch."""
        async with self.switch_lock:
            pass
        if is_model_repo(requested) and requested != self.current_model_name:
            await self.switch_model(requested)
        return self.model_epoch

    async def shutdown(self) -> None:
        log.info("Shutting down")
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self.executor, self.unload_blocking)
        finally:
            self.executor.shutdown(wait=False, cancel_futures=True)


runtime = ModelRuntime()
