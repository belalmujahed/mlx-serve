from types import SimpleNamespace

import pytest

from mlx_serve import catalog


def test_downloaded_models_filters_and_sorts_huggingface_cache(monkeypatch):
    repos = [
        SimpleNamespace(repo_type="model", repo_id="mlx-community/z-model"),
        SimpleNamespace(repo_type="model", repo_id="mlx-community/a-model"),
        SimpleNamespace(repo_type="dataset", repo_id="mlx-community/not-a-model"),
        SimpleNamespace(repo_type="model", repo_id="other/model"),
    ]
    monkeypatch.setattr(catalog, "scan_cache_dir", lambda: SimpleNamespace(repos=repos))

    assert catalog.downloaded_models() == [
        "mlx-community/a-model",
        "mlx-community/z-model",
    ]


def test_downloaded_models_returns_empty_on_cache_error(monkeypatch):
    def fail_scan():
        raise OSError("cache unavailable")

    monkeypatch.setattr(catalog, "scan_cache_dir", fail_scan)

    assert catalog.downloaded_models() == []


def test_refresh_discovery_filters_results_and_respects_limit(monkeypatch):
    class FakeHfApi:
        def list_models(self, **kwargs):
            assert kwargs == {
                "author": "mlx-community",
                "filter": "text-generation",
                "sort": "last_modified",
                "limit": 300,
            }
            return [
                SimpleNamespace(id="mlx-community/Qwen3-Coder-Instruct"),
                SimpleNamespace(id="mlx-community/GLM-Vision-Instruct"),
                SimpleNamespace(id="mlx-community/whisper-large"),
                SimpleNamespace(id="mlx-community/DeepSeek-R1"),
            ]

    monkeypatch.setattr(catalog, "HfApi", FakeHfApi)
    monkeypatch.setattr(catalog.config, "DISCOVER_MODELS", True)
    monkeypatch.setattr(catalog.config, "DISCOVER_LIMIT", 1)
    monkeypatch.setattr(catalog.time, "time", lambda: 123.0)

    catalog.refresh_discovery()

    assert catalog._discovered == ["mlx-community/Qwen3-Coder-Instruct"]
    assert catalog._discovered_at == 123.0


def test_refresh_discovery_disabled_leaves_cached_values(monkeypatch):
    monkeypatch.setattr(catalog.config, "DISCOVER_MODELS", False)
    monkeypatch.setattr(catalog, "_discovered", ["cached"])
    monkeypatch.setattr(catalog, "_discovered_at", 42.0)

    catalog.refresh_discovery()

    assert catalog._discovered == ["cached"]
    assert catalog._discovered_at == 42.0


def test_refresh_discovery_logs_failure_and_updates_refresh_time(monkeypatch, caplog):
    class FailedHfApi:
        def list_models(self, **kwargs):
            raise RuntimeError("offline")

    monkeypatch.setattr(catalog, "HfApi", FailedHfApi)
    monkeypatch.setattr(catalog.config, "DISCOVER_MODELS", True)
    monkeypatch.setattr(catalog.time, "time", lambda: 456.0)

    with caplog.at_level("WARNING", logger="proxy"):
        catalog.refresh_discovery()

    assert "Model discovery failed: offline" in caplog.text
    assert catalog._discovered_at == 456.0


def test_discover_models_starts_one_background_refresh_when_stale(monkeypatch):
    started = []

    class FakeThread:
        def __init__(self, target, daemon):
            assert daemon is True
            self.target = target

        def start(self):
            started.append(self.target)

    monkeypatch.setattr(catalog, "_discovered", ["stale"])
    monkeypatch.setattr(catalog, "_discovered_at", 0.0)
    monkeypatch.setattr(catalog.time, "time", lambda: 5000.0)
    monkeypatch.setattr(catalog.threading, "Thread", FakeThread)

    assert catalog.discover_models() == ["stale"]
    assert started == [catalog.refresh_discovery]
    assert catalog._discovered_at == 5000.0


def test_discover_models_does_not_refresh_fresh_cache(monkeypatch):
    monkeypatch.setattr(catalog, "_discovered", ["fresh"])
    monkeypatch.setattr(catalog, "_discovered_at", 5000.0)
    monkeypatch.setattr(catalog.time, "time", lambda: 5001.0)

    assert catalog.discover_models() == ["fresh"]


def test_list_catalog_orders_and_deduplicates_sources(monkeypatch):
    monkeypatch.setattr(catalog, "downloaded_models", lambda: ["mlx-community/local"])
    monkeypatch.setattr(catalog, "discover_models", lambda: ["mlx-community/curated"])
    monkeypatch.setattr(
        catalog.config,
        "CURATED_MODELS",
        ["mlx-community/curated", "mlx-community/another"],
    )

    result = catalog.list_catalog("mlx-community/local")

    assert result == [
        {"id": "mlx-community/local", "state": "loaded"},
        {"id": "mlx-community/curated", "state": "will download"},
        {"id": "mlx-community/another", "state": "will download"},
    ]


def test_list_catalog_without_current_model(monkeypatch):
    monkeypatch.setattr(catalog, "downloaded_models", lambda: [])
    monkeypatch.setattr(catalog, "discover_models", lambda: [])
    monkeypatch.setattr(catalog.config, "CURATED_MODELS", [])

    assert catalog.list_catalog(None) == []


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("org/model", True),
        ("org/model-name_1", True),
        ("claude-sonnet", False),
        ("org/model/extra", False),
        ("", False),
    ],
)
def test_is_model_repo(name, expected):
    assert catalog.is_model_repo(name) is expected
