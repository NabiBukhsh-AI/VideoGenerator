"""Exception hierarchy for videogen.

Every failure the pipeline can produce is one of these, so callers (CLI, Streamlit
UI, library users) can catch `VideoGenError` and render something useful instead of
a traceback from deep inside a vendor SDK.
"""

from __future__ import annotations


class VideoGenError(Exception):
    """Base class for every error raised by this package."""

    #: Short, user-facing hint on how to fix it. Rendered by the CLI/UI.
    hint: str | None = None

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if hint is not None:
            self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.hint:
            return f"{self.message}\nHint: {self.hint}"
        return self.message


class ConfigError(VideoGenError):
    """Configuration is missing or contradictory (e.g. no API key for a provider)."""


class DependencyError(VideoGenError):
    """An optional dependency or external binary is missing."""


class IngestError(VideoGenError):
    """Source material could not be read or contained no usable text."""


class ScriptError(VideoGenError):
    """The LLM did not return a script we can work with."""


class ProviderError(VideoGenError):
    """A provider (LLM / image / TTS) call failed."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retryable: bool = False,
        hint: str | None = None,
    ) -> None:
        super().__init__(message, hint=hint)
        self.provider = provider
        self.retryable = retryable


class RenderError(VideoGenError):
    """Frame generation, encoding or muxing failed."""


class FFmpegError(RenderError):
    """An ffmpeg/ffprobe invocation returned a non-zero exit code."""

    def __init__(self, message: str, *, command: list[str] | None = None, stderr: str = "") -> None:
        super().__init__(message)
        self.command = command or []
        self.stderr = stderr
