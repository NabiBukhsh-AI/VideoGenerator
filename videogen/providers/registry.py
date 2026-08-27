"""Provider factories.

One place that maps a config string to a concrete provider. Imports are deferred into
the factory bodies so that installing only, say, the ``elevenlabs`` extra does not make
``import videogen`` fail on a missing ``anthropic``.
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import Settings
from ..errors import ConfigError
from ..logging import get_logger
from .base import ImageProvider, LLMProvider, TTSProvider

log = get_logger("providers.registry")


def build_llm(settings: Settings, override: str | None = None) -> LLMProvider:
    name = (override or settings.llm_provider).lower()

    if name == "openai":
        from .llm.openai_llm import OpenAILLM

        return OpenAILLM(settings)
    if name == "anthropic":
        from .llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM(settings)
    if name in {"echo", "offline", "none"}:
        from .llm.echo_llm import EchoLLM

        return EchoLLM(settings)

    raise ConfigError(
        f"Unknown LLM provider {name!r}.",
        hint="Valid values: openai, anthropic, echo.",
    )


def build_image(settings: Settings, override: str | None = None) -> ImageProvider:
    name = (override or settings.image_provider).lower()

    if name == "openai":
        from .image.openai_image import OpenAIImage

        return OpenAIImage(settings)
    if name in {"placeholder", "offline", "none"}:
        from .image.placeholder_image import PlaceholderImage

        return PlaceholderImage()

    raise ConfigError(
        f"Unknown image provider {name!r}.",
        hint="Valid values: openai, placeholder.",
    )


def build_tts(settings: Settings, override: str | None = None) -> TTSProvider:
    name = (override or settings.tts_provider).lower()

    if name == "elevenlabs":
        from .tts.elevenlabs_tts import ElevenLabsTTS

        return ElevenLabsTTS(settings)
    if name == "openai":
        from .tts.openai_tts import OpenAITTS

        return OpenAITTS(settings)
    if name in {"silent", "offline", "none"}:
        from .tts.silent_tts import SilentTTS

        return SilentTTS()

    raise ConfigError(
        f"Unknown TTS provider {name!r}.",
        hint="Valid values: elevenlabs, openai, silent.",
    )


#: Introspection for the CLI's ``providers`` command and the Streamlit sidebar.
KNOWN: dict[str, tuple[str, ...]] = {
    "llm": ("openai", "anthropic", "echo"),
    "image": ("openai", "placeholder"),
    "tts": ("elevenlabs", "openai", "silent"),
}

BUILDERS: dict[str, Callable[..., object]] = {
    "llm": build_llm,
    "image": build_image,
    "tts": build_tts,
}
