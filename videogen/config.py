"""Configuration.

Secrets come from the environment or a local ``.env`` file and are held as
``SecretStr`` so they do not leak into logs, tracebacks or ``repr()`` output.

The previous version hardcoded live OpenAI and ElevenLabs keys in three source files,
which published them the moment the repo went up. Nothing here ever accepts a key as a
literal, and ``.env`` is gitignored.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigError

LLMProviderName = Literal["openai", "anthropic", "echo"]
ImageProviderName = Literal["openai", "placeholder"]
TTSProviderName = Literal["elevenlabs", "openai", "silent"]


class Settings(BaseSettings):
    """Runtime settings, resolved from env vars / ``.env`` / defaults."""

    model_config = SettingsConfigDict(
        env_prefix="VIDEOGEN_",
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Credentials -----------------------------------------------------------
    # Aliased to the conventional names so an existing OPENAI_API_KEY just works.
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    elevenlabs_api_key: SecretStr | None = Field(
        default=None, validation_alias="ELEVENLABS_API_KEY"
    )

    # --- Provider selection ----------------------------------------------------
    llm_provider: LLMProviderName = "openai"
    image_provider: ImageProviderName = "openai"
    tts_provider: TTSProviderName = "elevenlabs"

    # --- Model choices ---------------------------------------------------------
    openai_model: str = "gpt-4.1"
    anthropic_model: str = "claude-sonnet-4-5"
    image_model: str = "gpt-image-1"
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "alloy"
    elevenlabs_model: str = "eleven_multilingual_v2"
    elevenlabs_voice: str = "Rachel"

    # --- Generation defaults ---------------------------------------------------
    max_scenes: int = Field(default=6, ge=1, le=30)
    min_scenes: int = Field(default=3, ge=1, le=30)
    max_source_words: int = Field(default=6000, ge=100)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    image_style: str = "cinematic photography, dramatic lighting, high detail"

    # --- Execution -------------------------------------------------------------
    output_dir: Path = Path("output")
    max_workers: int = Field(default=4, ge=1, le=16)
    request_timeout_s: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=4, ge=1, le=10)
    cache_enabled: bool = True
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    font_path: Path | None = None
    log_level: str = "INFO"

    # --- Alignment -------------------------------------------------------------
    # "whisper" buys real per-word caption timing at the cost of one extra API call
    # per scene; "heuristic" is free and weights words by length.
    aligner: Literal["heuristic", "whisper"] = "heuristic"
    whisper_model: str = "whisper-1"

    @field_validator("min_scenes")
    @classmethod
    def _min_le_max(cls, value: int, info) -> int:
        max_scenes = info.data.get("max_scenes")
        if max_scenes is not None and value > max_scenes:
            raise ValueError(f"min_scenes ({value}) cannot exceed max_scenes ({max_scenes})")
        return value

    @field_validator("output_dir", mode="before")
    @classmethod
    def _expand(cls, value):
        return Path(str(value)).expanduser() if value else value

    # --- Convenience -----------------------------------------------------------
    def secret(self, name: str) -> str | None:
        """Return a credential's plain value, or None when unset."""
        value = getattr(self, name, None)
        return value.get_secret_value() if isinstance(value, SecretStr) else None

    def require(self, name: str, *, provider: str) -> str:
        """Return a credential or raise a ConfigError naming how to supply it."""
        value = self.secret(name)
        if not value:
            env_name = name.upper()
            raise ConfigError(
                f"{provider} requires {env_name}, which is not set.",
                hint=(
                    f"Add {env_name}=... to your .env file, or export it in your shell. "
                    "See .env.example for the full list."
                ),
            )
        return value

    def available_providers(self) -> dict[str, list[str]]:
        """Which providers can actually run given the credentials present.

        Used by the CLI ``doctor`` command and the Streamlit sidebar so users see why
        a provider is unselectable instead of hitting an auth error mid-run.
        """
        has_openai = bool(self.secret("openai_api_key"))
        return {
            "llm": (
                (["openai"] if has_openai else [])
                + (["anthropic"] if self.secret("anthropic_api_key") else [])
                + ["echo"]
            ),
            "image": (["openai"] if has_openai else []) + ["placeholder"],
            "tts": (
                (["elevenlabs"] if self.secret("elevenlabs_api_key") else [])
                + (["openai"] if has_openai else [])
                + ["silent"]
            ),
        }

    def offline(self) -> Settings:
        """A copy that uses only local providers - no network, no spend."""
        return self.model_copy(
            update={
                "llm_provider": "echo",
                "image_provider": "placeholder",
                "tts_provider": "silent",
            }
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()


def reload_settings() -> Settings:
    """Drop the cached settings and re-read the environment (used by tests and the UI)."""
    get_settings.cache_clear()
    return get_settings()


def dotenv_path() -> Path:
    """Where the CLI writes ``.env`` when scaffolding a new checkout."""
    return Path(os.environ.get("VIDEOGEN_DOTENV", ".env")).expanduser()
