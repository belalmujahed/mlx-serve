import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

load_dotenv()


class Config(BaseModel):
    """Validated settings for the Claude Code MLX Proxy."""

    model_config = ConfigDict(allow_inf_nan=False)

    HOST: str = Field(default="127.0.0.1", min_length=1)
    PORT: int = Field(default=8888, ge=1, le=65535)

    MODEL_NAME: str = Field(default="mlx-community/GLM-4.5-Air-3bit", min_length=1)
    PRELOAD: bool = True
    TRUST_REMOTE_CODE: bool = False
    EOS_TOKEN: Optional[str] = None

    CURATED_MODELS: list[str] = Field(
        default_factory=lambda: [
            "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
            "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
            "mlx-community/GLM-4.5-Air-3bit",
            "mlx-community/gpt-oss-20b-MXFP4-Q8",
        ]
    )
    DISCOVER_MODELS: bool = True
    DISCOVER_LIMIT: int = Field(default=10, ge=0)

    SHOW_THINKING: bool = True

    DEFAULT_MAX_TOKENS: int = Field(default=4096, gt=0)
    DEFAULT_TEMPERATURE: float = Field(default=1.0, ge=0)
    DEFAULT_TOP_P: float = Field(default=1.0, ge=0, le=1)

    CACHE_ENTRIES: int = Field(default=3, ge=0)
    CACHE_MIN_REUSE: int = Field(default=256, ge=0)

    API_MODEL_NAME: str = "claude-4-sonnet-20250514"
    VERBOSE: bool = False

    @field_validator("CURATED_MODELS", mode="before")
    @classmethod
    def split_curated_models(cls, value: object) -> list[str]:
        if isinstance(value, str):
            value = value.split(",")
        if not isinstance(value, list) or not all(
            isinstance(model, str) for model in value
        ):
            raise ValueError("CURATED_MODELS must be a comma-separated string or list")
        return [model.strip() for model in value if model.strip()]

    @classmethod
    def from_environment(cls) -> "Config":
        values = {
            name: os.environ[name] for name in cls.model_fields if name in os.environ
        }
        return cls.model_validate(values)

    def apply_overrides(self, **overrides: object) -> None:
        """Validate and apply explicit settings (e.g. CLI flags) over the environment."""
        updated = type(self).model_validate(
            {
                **self.model_dump(),
                **{k: v for k, v in overrides.items() if v is not None},
            }
        )
        for name in type(self).model_fields:
            setattr(self, name, getattr(updated, name))


config = Config.from_environment()
