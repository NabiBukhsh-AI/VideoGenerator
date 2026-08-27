"""Offline LLM stub.

Produces a schema-valid script by extractive summarisation of the source text - no
network, no key, no cost. This is what powers ``--dry-run`` and the test suite, and it
is why the whole pipeline can be exercised in CI.

The output is deliberately unglamorous. It is a scaffold for verifying wiring and
render behaviour, not a substitute for a real model.
"""

from __future__ import annotations

import re
from typing import Any

from ...config import Settings
from ...models import Usage
from ...script.prompts import SOURCE_DELIMITER
from ...utils import normalize_text
from ..base import LLMProvider, LLMResponse

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Words that carry no topical signal when building an image prompt.
# Kept as a prose block rather than 40 string literals: easier to scan and edit.
_STOPWORDS = frozenset(
    (
        "a an and are as at be but by for from has have he her his in into is it its of on "
        "or she that the their them they this to was were will with you your our we us"
    ).split()
)


class EchoLLM(LLMProvider):
    name = "echo"
    offline = True

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.7,
    ) -> LLMResponse:
        source = _extract_source(user)
        sentences = _split_sentences(source)

        max_scenes = getattr(self.settings, "max_scenes", 6) if self.settings else 6
        min_scenes = getattr(self.settings, "min_scenes", 3) if self.settings else 3
        chosen = _pick_evenly(sentences, max_scenes) or ["No source text was provided."]
        while len(chosen) < min_scenes:
            chosen.append(chosen[len(chosen) % len(chosen)])

        style = getattr(self.settings, "image_style", "cinematic") if self.settings else "cinematic"
        scenes = [
            {"narration": sentence, "image_prompt": _image_prompt(sentence, style)}
            for sentence in chosen
        ]

        data = {
            "title": _title(sentences),
            "description": chosen[0][:200],
            "hashtags": _hashtags(source),
            "scenes": scenes,
        }
        return LLMResponse(data=data, raw_text="<echo>", usage=Usage(llm_calls=1))


def _extract_source(user: str) -> str:
    """Pull the source material out of the rendered user prompt."""
    if SOURCE_DELIMITER in user:
        user = user.split(SOURCE_DELIMITER, 1)[1]
    return normalize_text(user)


def _split_sentences(text: str) -> list[str]:
    sentences = []
    for raw in _SENTENCE_SPLIT.split(text):
        sentence = raw.strip()
        # Skip fragments too short to narrate and clip ones too long to fit a beat.
        if len(sentence.split()) < 4:
            continue
        words = sentence.split()
        if len(words) > 28:
            sentence = " ".join(words[:28]) + "."
        sentences.append(sentence)
    return sentences


def _pick_evenly(sentences: list[str], count: int) -> list[str]:
    """Sample sentences spread across the document rather than only the opening."""
    if not sentences:
        return []
    if len(sentences) <= count:
        return list(sentences)
    step = len(sentences) / count
    return [sentences[min(len(sentences) - 1, int(i * step))] for i in range(count)]


def _keywords(text: str, limit: int) -> list[str]:
    seen: list[str] = []
    for word in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", text):
        lowered = word.lower()
        if lowered in _STOPWORDS or lowered in seen:
            continue
        seen.append(lowered)
        if len(seen) >= limit:
            break
    return seen


def _image_prompt(sentence: str, style: str) -> str:
    subject = ", ".join(_keywords(sentence, 5)) or "abstract background"
    return f"{subject}, {style}, no text, no watermark"


def _title(sentences: list[str]) -> str:
    if not sentences:
        return "Untitled Short"
    words = _keywords(sentences[0], 5)
    return " ".join(word.capitalize() for word in words)[:120] or "Untitled Short"


def _hashtags(text: str) -> list[str]:
    return _keywords(text, 5)
