"""Turn source material into a validated `Script`."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import Settings
from ..errors import ScriptError
from ..logging import get_logger
from ..models import Scene, Script, SourceDocument, Usage
from ..providers.base import LLMProvider
from ..utils import normalize_text
from .prompts import SYSTEM_PROMPT, build_user_prompt, script_schema

log = get_logger("script")

# Speaker labels the model sometimes prefixes despite instructions.
_SPEAKER_PREFIX = re.compile(r"^\s*(narrator|voice ?over|vo|host|speaker)\s*[:\-]\s*", re.I)
# Stage directions in brackets, e.g. "[upbeat] Today we..." or "(pause)".
_STAGE_DIRECTION = re.compile(r"^\s*[\[(][^\])]{0,40}[\])]\s*")


@dataclass
class ScriptRequest:
    """Inputs to one script generation."""

    document: SourceDocument
    tone: str | None = None
    language: str | None = None
    focus: str | None = None


class ScriptGenerator:
    """Builds a `Script` from a `SourceDocument` using any `LLMProvider`."""

    def __init__(self, provider: LLMProvider, settings: Settings) -> None:
        self.provider = provider
        self.settings = settings

    def generate(self, request: ScriptRequest) -> tuple[Script, Usage]:
        document = request.document.truncate(self.settings.max_source_words)
        schema = script_schema(
            min_scenes=self.settings.min_scenes, max_scenes=self.settings.max_scenes
        )
        user_prompt = build_user_prompt(
            document.text,
            min_scenes=self.settings.min_scenes,
            max_scenes=self.settings.max_scenes,
            style=self.settings.image_style,
            tone=request.tone,
            language=request.language,
            focus=request.focus,
        )

        log.info(
            "Generating script from %d words via %s", document.word_count, self.provider.label
        )
        response = self.provider.complete_json(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            schema=schema,
            temperature=self.settings.temperature,
        )

        script = self._build(response.data, document)
        log.info(
            "Script %r: %d scenes, ~%.0fs estimated",
            script.title,
            len(script.scenes),
            script.estimated_duration_ms / 1000,
        )
        return script, response.usage

    # -- internals -------------------------------------------------------------
    def _build(self, data: dict, document: SourceDocument) -> Script:
        raw_scenes = data.get("scenes")
        if not isinstance(raw_scenes, list) or not raw_scenes:
            raise ScriptError(
                "The model returned no scenes.",
                hint="Try re-running, or switch model with --llm.",
            )

        scenes: list[Scene] = []
        for index, raw in enumerate(raw_scenes):
            if not isinstance(raw, dict):
                continue
            narration = _clean_narration(str(raw.get("narration", "")))
            image_prompt = normalize_text(str(raw.get("image_prompt", "")))

            if not narration:
                log.warning("Scene %d has empty narration; dropping it.", index + 1)
                continue
            if not image_prompt:
                # A missing image prompt is recoverable - derive one from the narration
                # rather than losing the scene entirely.
                image_prompt = f"{narration} {self.settings.image_style}"
                log.warning(
                    "Scene %d had no image prompt; derived one from the narration.", index + 1
                )

            scenes.append(
                Scene(index=len(scenes), narration=narration, image_prompt=image_prompt)
            )

        if not scenes:
            raise ScriptError("Every scene the model returned was unusable.")

        if len(scenes) > self.settings.max_scenes:
            log.info(
                "Model returned %d scenes; trimming to %d.",
                len(scenes),
                self.settings.max_scenes,
            )
            scenes = scenes[: self.settings.max_scenes]

        title = normalize_text(str(data.get("title") or "")) or _fallback_title(document, scenes)

        return Script(
            title=title[:120],
            scenes=scenes,
            description=normalize_text(str(data.get("description") or "")),
            hashtags=data.get("hashtags") or [],
            source_name=document.name,
        ).renumber()


def _clean_narration(text: str) -> str:
    """Strip speaker labels, stage directions and wrapping quotes."""
    text = normalize_text(text)
    text = _SPEAKER_PREFIX.sub("", text)
    text = _STAGE_DIRECTION.sub("", text)
    text = text.strip().strip('"').strip("'").strip()
    return text


def _fallback_title(document: SourceDocument, scenes: list[Scene]) -> str:
    if document.metadata.get("title"):
        return str(document.metadata["title"])
    if scenes:
        words = scenes[0].narration.split()[:8]
        return " ".join(words).rstrip(".,;:") or document.name
    return document.name or "Untitled Short"
