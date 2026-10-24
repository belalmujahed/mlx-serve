import os

import pytest
from pydantic import ValidationError

from mlx_serve.config import Config


def test_config_parses_environment_values(monkeypatch):
    monkeypatch.setattr(
        os,
        "environ",
        {
            "PORT": "9000",
            "TRUST_REMOTE_CODE": "true",
            "CURATED_MODELS": "org/one, org/two ,,",
        },
    )

    config = Config.from_environment()

    assert config.PORT == 9000
    assert config.TRUST_REMOTE_CODE is True
    assert config.CURATED_MODELS == ["org/one", "org/two"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("PORT", "0"),
        ("PORT", "not-a-port"),
        ("HOST", ""),
        ("DEFAULT_MAX_TOKENS", "0"),
        ("DEFAULT_TOP_P", "1.5"),
        ("DEFAULT_TEMPERATURE", "nan"),
        ("CACHE_ENTRIES", "-1"),
    ],
)
def test_config_rejects_invalid_values(field, value):
    with pytest.raises(ValidationError):
        Config.model_validate({field: value})


def test_config_rejects_invalid_curated_model_list():
    with pytest.raises(ValidationError):
        Config.model_validate({"CURATED_MODELS": 42})


def test_apply_overrides_validates_and_skips_none():
    config = Config()

    config.apply_overrides(MODEL_NAME="org/model", PORT=None, PRELOAD=False)

    assert config.MODEL_NAME == "org/model"
    assert config.PORT == 8888
    assert config.PRELOAD is False
    with pytest.raises(ValidationError):
        config.apply_overrides(PORT=0)
    assert config.PORT == 8888
