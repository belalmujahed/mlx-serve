import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mlx_serve import runtime as runtime_module
from mlx_serve.runtime import ModelRuntime


def test_shutdown_unloads_model_before_shutting_down_executor(monkeypatch):
    runtime = ModelRuntime()
    unloaded = []
    monkeypatch.setattr(runtime, "unload_blocking", lambda: unloaded.append(True))

    asyncio.run(runtime.shutdown())

    assert unloaded == [True]


def test_model_loader_returns_standard_loader_result(monkeypatch):
    runtime = ModelRuntime()
    loaded = (object(), object())
    monkeypatch.setattr(runtime_module, "load", lambda name, tokenizer_config: loaded)

    assert runtime._load_model_with_fallback("org/model", {}) == loaded


def test_model_loader_uses_fast_tokenizer_fallback(monkeypatch):
    runtime = ModelRuntime()
    mlx_model = object()
    hf_tokenizer = object()
    wrapped_tokenizer = object()
    monkeypatch.setattr(
        runtime_module,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("TokenizersBackend unavailable")
        ),
    )
    monkeypatch.setattr(
        runtime_module, "get_model_path", lambda name: Path("/models/test")
    )
    monkeypatch.setattr(
        runtime_module, "load_model", lambda path: (mlx_model, {"eos_token_id": 2})
    )

    class FakeTokenizerLoader:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert path == "/models/test"
            assert kwargs == {"trust_remote_code": True}
            return hf_tokenizer

    monkeypatch.setattr(runtime_module, "PreTrainedTokenizerFast", FakeTokenizerLoader)
    monkeypatch.setattr(
        runtime_module,
        "TokenizerWrapper",
        lambda tokenizer, eos_token_ids: (
            wrapped_tokenizer
            if tokenizer is hf_tokenizer and eos_token_ids == 2
            else None
        ),
    )

    assert runtime._load_model_with_fallback(
        "org/model", {"trust_remote_code": True}
    ) == (mlx_model, wrapped_tokenizer)


def test_model_loader_reraises_unrelated_value_error(monkeypatch):
    runtime = ModelRuntime()

    def fail(*args, **kwargs):
        raise ValueError("different loader problem")

    monkeypatch.setattr(runtime_module, "load", fail)

    with pytest.raises(ValueError, match="different loader problem"):
        runtime._load_model_with_fallback("org/model", {})


def test_load_blocking_applies_tokenizer_config_and_warms_up(monkeypatch):
    runtime = ModelRuntime()
    loaded = (object(), object())
    captured = {}
    monkeypatch.setattr(runtime_module.config, "TRUST_REMOTE_CODE", True)
    monkeypatch.setattr(runtime_module.config, "EOS_TOKEN", "</s>")
    monkeypatch.setattr(
        runtime,
        "_load_model_with_fallback",
        lambda name, tokenizer_config: (
            captured.update(name=name, config=tokenizer_config) or loaded
        ),
    )
    warmed = []
    monkeypatch.setattr(runtime, "warmup", lambda: warmed.append(True))

    runtime.load_blocking("org/model")

    assert captured == {
        "name": "org/model",
        "config": {"trust_remote_code": True, "eos_token": "</s>"},
    }
    assert (runtime.model, runtime.tokenizer, runtime.current_model_name) == (
        loaded[0],
        loaded[1],
        "org/model",
    )
    assert warmed == [True]


def test_load_blocking_records_failure_reason_and_reraises(monkeypatch):
    runtime = ModelRuntime()

    def fail(name, tokenizer_config):
        raise OSError("download failed")

    monkeypatch.setattr(runtime, "_load_model_with_fallback", fail)

    assert runtime.load_failure_reason is None
    with pytest.raises(OSError, match="download failed"):
        runtime.load_blocking("org/model")

    assert runtime.load_failure_reason == "download failed"
    assert runtime.model is None


def test_unload_clears_model_caches_and_mlx_memory(monkeypatch):
    runtime = ModelRuntime()
    runtime.model = object()
    runtime.tokenizer = object()
    runtime.current_model_name = "org/model"
    runtime.prompt_caches.append({"cache": [], "tokens": []})
    calls = []
    monkeypatch.setattr(runtime_module.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(runtime_module.mx, "clear_cache", lambda: calls.append("mlx"))

    runtime.unload_blocking()

    assert runtime.model is None
    assert runtime.tokenizer is None
    assert runtime.current_model_name is None
    assert runtime.prompt_caches == []
    assert calls == ["gc", "mlx"]


def test_warmup_evaluates_parameters_and_generates_short_prompt(monkeypatch):
    runtime = ModelRuntime()
    runtime.model = SimpleNamespace(parameters=lambda: "weights")
    runtime.tokenizer = SimpleNamespace(vocab_size=8)
    evaluated = []
    generated = []
    monkeypatch.setattr(
        runtime_module.mx, "eval", lambda value: evaluated.append(value)
    )
    monkeypatch.setattr(
        runtime_module,
        "stream_generate",
        lambda model, tokenizer, **kwargs: (
            generated.append((model, tokenizer, kwargs)) or iter([None])
        ),
    )

    runtime.warmup()

    assert evaluated == ["weights"]
    assert generated[0][2] == {
        "prompt": [(i * 7919) % 8 for i in range(1024)],
        "max_tokens": 4,
    }


def test_switch_model_skips_reload_when_model_is_already_active(monkeypatch):
    runtime = ModelRuntime()
    runtime.current_model_name = "org/model"
    calls = []
    monkeypatch.setattr(runtime, "unload_blocking", lambda: calls.append("unload"))
    monkeypatch.setattr(
        runtime, "load_blocking", lambda name: pytest.fail("should not reload")
    )

    async def run():
        await runtime.switch_model("org/model")
        await runtime.shutdown()

    asyncio.run(run())
    assert calls == ["unload"]


def test_switch_model_loads_requested_model(monkeypatch):
    runtime = ModelRuntime()
    calls = []
    monkeypatch.setattr(runtime, "unload_blocking", lambda: calls.append("unload"))
    monkeypatch.setattr(runtime, "load_blocking", lambda name: calls.append(name))

    async def run():
        await runtime.switch_model("org/model")
        await runtime.shutdown()

    asyncio.run(run())

    assert calls == ["unload", "org/model", "unload"]
    assert runtime.model_epoch == 1


def test_switch_model_restores_previous_model_after_load_failure(monkeypatch):
    runtime = ModelRuntime()
    runtime.current_model_name = "org/previous"
    calls = []

    def load(name):
        calls.append(name)
        if name == "org/new":
            raise RuntimeError("load failed")

    monkeypatch.setattr(runtime, "unload_blocking", lambda: None)
    monkeypatch.setattr(runtime, "load_blocking", load)

    async def run():
        with pytest.raises(HTTPException, match="Could not load model org/new"):
            await runtime.switch_model("org/new")
        await runtime.shutdown()

    asyncio.run(run())

    assert calls == ["org/new", "org/previous"]


def test_switch_model_failure_without_previous_model_raises_http_error(monkeypatch):
    runtime = ModelRuntime()
    monkeypatch.setattr(runtime, "unload_blocking", lambda: None)

    def fail(name):
        raise RuntimeError("load failed")

    monkeypatch.setattr(runtime, "load_blocking", fail)

    async def run():
        with pytest.raises(HTTPException, match="Could not load model org/new"):
            await runtime.switch_model("org/new")
        await runtime.shutdown()

    asyncio.run(run())


def test_resolve_model_switches_only_for_new_hub_repo(monkeypatch):
    runtime = ModelRuntime()
    calls = []
    monkeypatch.setattr(runtime, "unload_blocking", lambda: None)
    monkeypatch.setattr(runtime, "switch_model", lambda name: async_append(calls, name))

    async def resolve():
        assert await runtime.resolve_model("claude-alias") == 0
        assert await runtime.resolve_model("org/model") == 0
        runtime.current_model_name = "org/model"
        assert await runtime.resolve_model("org/model") == 0
        await runtime.shutdown()

    def async_append(items, value):
        async def append():
            items.append(value)

        return append()

    asyncio.run(resolve())

    assert calls == ["org/model"]


def test_shutdown_shuts_executor_even_when_unload_fails(monkeypatch):
    runtime = ModelRuntime()
    monkeypatch.setattr(
        runtime,
        "unload_blocking",
        lambda: (_ for _ in ()).throw(RuntimeError("unload failed")),
    )

    with pytest.raises(RuntimeError, match="unload failed"):
        asyncio.run(runtime.shutdown())

    assert runtime.executor._shutdown


def test_get_model_path_prefers_local_directory(tmp_path):
    assert runtime_module.get_model_path(str(tmp_path)) == tmp_path


def test_get_model_path_downloads_repository(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runtime_module,
        "snapshot_download",
        lambda name, allow_patterns: calls.append((name, allow_patterns)) or "/cache/m",
    )

    assert runtime_module.get_model_path("org/model") == Path("/cache/m")
    assert calls == [("org/model", runtime_module.DEFAULT_ALLOW_PATTERNS)]
