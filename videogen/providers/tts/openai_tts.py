"""OpenAI text-to-speech."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...config import Settings
from ...errors import DependencyError, ProviderError
from ...utils import retry
from ..base import TTSProvider

_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]
_PRICE_PER_1K_CHARS = 0.015


class OpenAITTS(TTSProvider):
    name = "openai"
    audio_format = "mp3"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.openai_tts_model
        self.default_voice = settings.openai_tts_voice
        self._client: Any = None

    def health_check(self) -> None:
        self.settings.require("openai_api_key", provider="OpenAI TTS")
        self._ensure_client()

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise DependencyError(
                "The openai package is not installed.",
                hint="pip install 'videogen[openai]'",
            ) from exc
        self._client = OpenAI(
            api_key=self.settings.require("openai_api_key", provider="OpenAI TTS"),
            timeout=self.settings.request_timeout_s,
            max_retries=0,
        )
        return self._client

    def list_voices(self) -> list[str]:
        return list(_VOICES)

    def synthesize(self, *, text: str, output_path: Path, voice: str | None = None) -> Path:
        return self._synthesize(text=text, output_path=output_path, voice=voice)

    @retry(attempts=4)
    def _synthesize(self, *, text: str, output_path: Path, voice: str | None) -> Path:
        client = self._ensure_client()
        chosen = (voice or self.default_voice or "alloy").strip().lower()
        if chosen not in _VOICES:
            chosen = "alloy"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=chosen,
                input=text,
                response_format="mp3",
            ) as response:
                response.stream_to_file(output_path)
        except Exception as exc:
            raise _wrap(exc) from exc

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise ProviderError(
                "OpenAI TTS produced an empty file.", provider=self.name, retryable=True
            )
        return output_path

    def estimate_cost(self, characters: int) -> float:
        return characters / 1000 * _PRICE_PER_1K_CHARS


def _wrap(exc: Exception) -> ProviderError:
    kind = type(exc).__name__
    transient = kind in {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
    }
    hint = "Check OPENAI_API_KEY." if kind == "AuthenticationError" else None
    return ProviderError(
        f"OpenAI TTS {kind}: {exc}", provider="openai", retryable=transient, hint=hint
    )
