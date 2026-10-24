from collections.abc import AsyncIterator
from itertools import count

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from mlx_serve import api
from mlx_serve.api import create_api_router
from mlx_serve.models import (
    ContentBlockText,
    MessageResponse,
    MessagesRequest,
    Usage,
)


class FakeRuntime:
    def __init__(self, loaded=True):
        self.request_ids = count(1)
        self.model = object() if loaded else None
        self.tokenizer = object() if loaded else None
        self.current_model_name = "mlx-community/test"
        self.benchmarks = [
            {
                "model": self.current_model_name,
                "ttft_s": 1.0,
                "prefill_tps": 10.0,
                "decode_tps": 20.0,
                "peak_memory_gb": 3.0,
            }
        ]
        self.benchmark_totals = {
            "generations": 1,
            "prompt_tokens": 2,
            "output_tokens": 3,
        }
        self.resolved_models = []

    async def switch_model(self, name):
        if name == "org/broken":
            raise HTTPException(status_code=502, detail="Could not load model")
        self.current_model_name = name
        self.model = object()

    async def resolve_model(self, model):
        self.resolved_models.append(model)
        return 4


def make_client(runtime=None, generate_error=False, format_error=False):
    runtime = runtime or FakeRuntime()
    calls = {}

    def format_messages(tokenizer, messages, system, tools, thinking):
        if format_error:
            raise ValueError("formatting failed")
        calls["formatted"] = (messages, system, tools, thinking)
        return "prompt"

    def encode_prompt(tokenizer, prompt):
        calls["encoded"] = prompt
        return [11, 12]

    async def generate_response(request, prompt_ids):
        if generate_error:
            raise ValueError("generation failed")
        calls["request_id"] = request._request_id
        return MessageResponse(
            id="msg_test",
            content=[ContentBlockText(text="response")],
            model=request.model,
            usage=Usage(input_tokens=len(prompt_ids), output_tokens=1),
        )

    async def stream_generate_response(request, prompt_ids) -> AsyncIterator[str]:
        yield "event: message_stop\ndata: {}\n\n"

    app = FastAPI()
    app.include_router(
        create_api_router(
            runtime,
            format_messages,
            encode_prompt,
            generate_response,
            stream_generate_response,
        )
    )
    return TestClient(app), runtime, calls


def message_payload(**overrides):
    return {
        "model": "mlx-community/test",
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "hello"}],
        **overrides,
    }


def test_message_endpoint_returns_response_and_sets_runtime_epoch():
    client, runtime, calls = make_client()

    response = client.post("/v1/messages", json=message_payload())

    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "response"}]
    assert response.json()["usage"] == {"input_tokens": 2, "output_tokens": 1}
    assert runtime.resolved_models == ["mlx-community/test"]
    assert calls["request_id"] == 1
    assert calls["encoded"] == "prompt"


def test_internal_request_fields_are_not_serialized():
    request = MessagesRequest.model_validate(message_payload())
    request._request_id = 7
    request._model_epoch = 4

    assert "request_id" not in request.model_dump()
    assert "model_epoch" not in request.model_dump()


def test_message_endpoint_streams_events():
    client, _, _ = make_client()

    response = client.post("/v1/messages", json=message_payload(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: message_stop" in response.text


def test_count_tokens_endpoint_uses_prompt_encoder():
    client, _, calls = make_client()

    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "mlx-community/test",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 2}
    assert calls["encoded"] == "prompt"


def test_count_tokens_endpoint_translates_format_exception():
    client, _, _ = make_client(format_error=True)

    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "mlx-community/test",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "formatting failed"


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens"])
def test_generation_endpoints_reject_requests_without_loaded_model(path):
    client, _, _ = make_client(FakeRuntime(loaded=False))
    payload = (
        message_payload()
        if path == "/v1/messages"
        else {"model": "mlx-community/test", "messages": []}
    )

    response = client.post(path, json=payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded"


def test_message_endpoint_translates_generation_exception():
    client, _, _ = make_client(generate_error=True)

    response = client.post("/v1/messages", json=message_payload())

    assert response.status_code == 500
    assert response.json()["detail"] == "generation failed"


def test_model_catalog_health_and_stats_endpoints(monkeypatch):
    monkeypatch.setattr(
        api,
        "list_catalog",
        lambda current: [{"id": current, "state": "loaded"}],
    )
    client, _, _ = make_client()

    catalog = client.get("/v1/models")
    health = client.get("/health")
    stats = client.get("/stats")
    model = client.get("/v1/models/org/model")
    root = client.get("/")

    assert catalog.status_code == 200
    assert [entry["id"] for entry in catalog.json()["data"]] == [
        "mlx-community/test",
        api.config.API_MODEL_NAME,
    ]
    assert health.json() == {
        "status": "healthy",
        "model_loaded": True,
        "model": "mlx-community/test",
    }
    assert stats.json()["avg_decode_tps"] == 20.0
    assert model.json()["id"] == "org/model"
    assert root.json() == {
        "message": "Claude Code MLX Proxy",
        "status": "running",
        "model_loaded": True,
    }


def test_tool_result_content_accepts_supported_json_shapes():
    from mlx_serve.models import ContentBlockToolResult

    content = [
        {"type": "text", "text": "result"},
        "extra",
        42,
    ]
    block = ContentBlockToolResult(tool_use_id="toolu_test", content=content)

    assert block.content == content


def test_model_listing_honors_limit(monkeypatch):
    monkeypatch.setattr(
        api,
        "list_catalog",
        lambda current: [
            {"id": "org/a", "state": "loaded"},
            {"id": "org/b", "state": "will download"},
        ],
    )
    client, _, _ = make_client()

    body = client.get("/v1/models?limit=1").json()

    assert [entry["id"] for entry in body["data"]] == ["org/a"]
    assert body["has_more"] is True
    assert client.get("/v1/models?limit=0").status_code == 422


def test_current_model_endpoint_reports_state():
    client, _, _ = make_client(FakeRuntime(loaded=False))

    assert client.get("/v1/model").json() == {
        "model": "mlx-community/test",
        "loaded": False,
    }


def test_switch_model_endpoint_switches_and_validates():
    client, runtime, _ = make_client(FakeRuntime(loaded=False))

    ok = client.post("/v1/model", json={"model": "org/new"})
    bad = client.post("/v1/model", json={"model": "claude-sonnet"})
    broken = client.post("/v1/model", json={"model": "org/broken"})

    assert ok.json() == {"model": "org/new", "loaded": True}
    assert runtime.current_model_name == "org/new"
    assert bad.status_code == 400
    assert broken.status_code == 502
