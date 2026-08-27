"""ElevenLabs text-to-speech.

Written against the v1+ SDK. The original code used ``set_api_key`` / ``generate`` /
``save``, which were module-level functions removed in ElevenLabs 1.0 - that import
alone breaks on any current install.

Voices are resolved by name to an ID once per process, so users can keep writing
"Rachel" instead of looking up a 20-character identifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...config import Settings
from ...errors import DependencyError, ProviderError
from ...logging import get_logger
from ...utils import retry
from ..base import TTSProvider

log = get_logger("providers.tts.elevenlabs")

# USD per 1000 characters on a mid-tier plan; indicative only.
_PRICE_PER_1K_CHARS = 0.18


class ElevenLabsTTS(TTSProvider):
    name = "elevenlabs"
    audio_format = "mp3"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.elevenlabs_model
        self.default_voice = settings.elevenlabs_voice
        self._client: Any = None
        self._voice_ids: dict[str, str] | None = None

    def health_check(self) -> None:
        self.settings.require("elevenlabs_api_key", provider="ElevenLabs")
        self._ensure_client()

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as exc:
            raise DependencyError(
                "The elevenlabs package is not installed (or is pre-1.0).",
                hint="pip install 'videogen[elevenlabs]'",
            ) from exc
        self._client = ElevenLabs(
            api_key=self.settings.require("elevenlabs_api_key", provider="ElevenLabs"),
            timeout=self.settings.request_timeout_s,
        )
        return self._client

    # -- voices ----------------------------------------------------------------
    def _voice_map(self) -> dict[str, str]:
        if self._voice_ids is not None:
            return self._voice_ids
        try:
            response = self._ensure_client().voices.get_all()
            self._voice_ids = {
                v.name.strip().lower(): v.voice_id for v in response.voices if v.name
            }
        except Exception as exc:
            log.warning("Could not list ElevenLabs voices: %s", exc)
            self._voice_ids = {}
        return self._voice_ids

    def list_voices(self) -> list[str]:
        return sorted(self._voice_map())

    def _resolve_voice(self, voice: str | None) -> str:
        wanted = (voice or self.default_voice or "").strip()
        if not wanted:
            raise ProviderError(
                "No ElevenLabs voice specified.",
                provider=self.name,
                hint="Set VIDEOGEN_ELEVENLABS_VOICE or pass --voice.",
            )
        # Already an ID (they are opaque 20-char strings with no spaces).
        if len(wanted) >= 20 and " " not in wanted:
            return wanted
        resolved = self._voice_map().get(wanted.lower())
        if resolved:
            return resolved
        log.warning("Voice %r not found in your library; sending it through as an ID.", wanted)
        return wanted

    # -- TTSProvider -----------------------------------------------------------
    def synthesize(self, *, text: str, output_path: Path, voice: str | None = None) -> Path:
        return self._synthesize(text=text, output_path=output_path, voice=voice)

    @retry(attempts=4)
    def _synthesize(self, *, text: str, output_path: Path, voice: str | None) -> Path:
        client = self._ensure_client()
        try:
            stream = client.text_to_speech.convert(
                voice_id=self._resolve_voice(voice),
                model_id=self.model,
                text=text,
                output_format="mp3_44100_128",
            )
            audio = b"".join(stream)
        except Exception as exc:
            raise _wrap(exc) from exc

        if not audio:
            raise ProviderError(
                "ElevenLabs returned empty audio.", provider=self.name, retryable=True
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio)
        return output_path

    def estimate_cost(self, characters: int) -> float:
        return characters / 1000 * _PRICE_PER_1K_CHARS


def _wrap(exc: Exception) -> ProviderError:
    kind = type(exc).__name__
    message = str(exc)
    transient = kind in {"RateLimitError", "ApiError", "TooManyRequestsError"} and (
        "429" in message or "rate" in message.lower() or "500" in message
    )
    hint = None
    if "quota" in message.lower():
        hint = "Your ElevenLabs character quota is exhausted; switch to --tts openai or top up."
        transient = False
    elif "401" in message or kind == "UnauthorizedError":
        hint = "Check ELEVENLABS_API_KEY. If this key was ever committed to git, rotate it."
    return ProviderError(
        f"ElevenLabs {kind}: {exc}", provider="elevenlabs", retryable=transient, hint=hint
    )
