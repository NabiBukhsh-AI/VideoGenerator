"""Provider interfaces.

Three capabilities the pipeline needs from the outside world: turn text into a script
(`LLMProvider`), turn a prompt into a picture (`ImageProvider`), and turn a sentence
into speech (`TTSProvider`).

Each is an ABC with a narrow surface. The pipeline depends only on these, so swapping
OpenAI for Anthropic, or any of them for an offline stub, changes one config value and
nothing else. Every concrete provider also has an offline sibling, which is what makes
`--dry-run` and the test suite work without credentials or network.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import Usage


@dataclass
class LLMResponse:
    """A structured completion plus whatever usage the vendor reported."""

    data: dict[str, Any]
    raw_text: str = ""
    usage: Usage = field(default_factory=Usage)


class Provider(ABC):  # noqa: B024 - marker base; subclasses declare the abstract API
    """Shared metadata every provider exposes."""

    #: Stable identifier used in config, cache keys and logs.
    name: str = "provider"

    #: False when the provider makes paid network calls.
    offline: bool = False

    @property
    def label(self) -> str:
        return f"{self.name}{' (offline)' if self.offline else ''}"

    def health_check(self) -> None:
        """Raise ConfigError/DependencyError if this provider cannot run.

        Called by ``videogen doctor`` and at pipeline start, so a missing key fails in
        the first second rather than after a minute of image generation.
        """
        return None


class LLMProvider(Provider):
    """Generates a JSON object conforming to a supplied schema."""

    @abstractmethod
    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Return a parsed JSON object matching ``schema``.

        Implementations must use the vendor's native structured-output mechanism where
        one exists, so the pipeline never has to parse prose. That replaces the old
        line-prefix parsing, which broke whenever the model reformatted its answer.
        """

    def transcribe_words(self, audio_path: Path) -> list[dict[str, Any]]:
        """Optional: word-level timestamps for ``audio_path``.

        Providers that cannot do this return an empty list and the caller falls back to
        heuristic alignment.
        """
        return []


class ImageProvider(Provider):
    """Renders a prompt to an image file on disk."""

    @abstractmethod
    def generate(
        self,
        *,
        prompt: str,
        output_path: Path,
        size: str = "1024x1536",
    ) -> Path:
        """Write an image to ``output_path`` and return it."""

    def estimate_cost(self, count: int) -> float:
        return 0.0


class TTSProvider(Provider):
    """Synthesises speech to an audio file on disk."""

    #: Container the provider writes. The renderer needs this to probe durations.
    audio_format: str = "mp3"

    @abstractmethod
    def synthesize(
        self,
        *,
        text: str,
        output_path: Path,
        voice: str | None = None,
    ) -> Path:
        """Write narration audio to ``output_path`` and return it."""

    def list_voices(self) -> list[str]:
        """Voices the user can pick from. Empty means free-form text entry."""
        return []

    def estimate_cost(self, characters: int) -> float:
        return 0.0
