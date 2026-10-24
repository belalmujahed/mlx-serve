import runpy
import sys

import pytest

from mlx_serve import main


@pytest.fixture(autouse=True)
def restore_config(monkeypatch):
    for name in type(main.config).model_fields:
        monkeypatch.setattr(main.config, name, getattr(main.config, name))


@pytest.fixture
def uvicorn_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        main.uvicorn,
        "run",
        lambda app, host, port: calls.append((app, host, port)),
    )
    return calls


def test_main_starts_uvicorn_with_configured_host_and_port(uvicorn_calls, capsys):
    main.main([])

    assert uvicorn_calls == [(main.app, main.config.HOST, main.config.PORT)]
    assert "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1" in capsys.readouterr().out


def test_cli_flags_override_configuration(uvicorn_calls):
    main.main(
        [
            "--model",
            "org/model",
            "--no-preload",
            "--host",
            "0.0.0.0",
            "--port",
            "9100",
            "--trust-remote-code",
            "--verbose",
        ]
    )

    assert main.config.MODEL_NAME == "org/model"
    assert main.config.PRELOAD is False
    assert main.config.TRUST_REMOTE_CODE is True
    assert main.config.VERBOSE is True
    assert uvicorn_calls == [(main.app, "0.0.0.0", 9100)]


def test_cli_rejects_invalid_port(uvicorn_calls):
    with pytest.raises(Exception, match="PORT"):
        main.main(["--port", "0"])
    assert uvicorn_calls == []


def test_module_entrypoint_calls_main(monkeypatch, uvicorn_calls):
    monkeypatch.setattr(sys, "argv", ["mlx-serve"])
    monkeypatch.delitem(sys.modules, "mlx_serve.main")

    runpy.run_module("mlx_serve.main", run_name="__main__")

    assert len(uvicorn_calls) == 1
