"""Word-level timing for captions.

The original split a line's duration evenly across its words, so "a" and
"extraordinarily" were highlighted for the same number of frames and the karaoke effect
drifted badly by the end of a sentence.

Two strategies here:

* `HeuristicAligner` weights each word by how long it takes to say - character count
  plus a pause for trailing punctuation. Free, instant, and good enough that the
  highlight tracks the voice.
* `WhisperAligner` asks a speech model for real timestamps. One extra API call per
  scene, and the highlight lands on the actual word.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from ..logging import get_logger
from ..models import Scene, WordTiming
from ..providers.base import LLMProvider

log = get_logger("render.align")

# Trailing punctuation implies a pause proportional to its weight.
_PAUSE_WEIGHT = {",": 1.6, ";": 2.0, ":": 2.0, ".": 2.6, "!": 2.6, "?": 2.8, "-": 1.2}

# Very short function words are spoken faster than their length suggests.
_MIN_WEIGHT = 1.8


class Aligner(ABC):
    """Assigns start/end times to each word of a scene's narration."""

    @abstractmethod
    def align(self, scene: Scene) -> list[WordTiming]:
        """Return timings covering ``scene.narration``, relative to the scene start."""


class HeuristicAligner(Aligner):
    """Duration-weighted alignment. No network, no cost."""

    def align(self, scene: Scene) -> list[WordTiming]:
        words = scene.narration.split()
        duration = scene.duration_ms or 0.0
        if not words or duration <= 0:
            return []

        weights = [_weight(word) for word in words]
        total = sum(weights) or float(len(words))

        timings: list[WordTiming] = []
        cursor = 0.0
        for word, weight in zip(words, weights, strict=True):
            span = duration * (weight / total)
            timings.append(
                WordTiming(word=word, start_ms=cursor, end_ms=min(duration, cursor + span))
            )
            cursor += span
        # Absorb rounding drift into the last word so timings end exactly on duration.
        if timings:
            timings[-1] = WordTiming(
                word=timings[-1].word, start_ms=timings[-1].start_ms, end_ms=duration
            )
        return timings


class WhisperAligner(Aligner):
    """Real timestamps from a speech-to-text model, with heuristic fallback."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self._fallback = HeuristicAligner()

    def align(self, scene: Scene) -> list[WordTiming]:
        if scene.audio_path is None or not scene.audio_path.exists():
            return self._fallback.align(scene)

        try:
            transcribed = self.provider.transcribe_words(scene.audio_path)
        except Exception as exc:
            log.warning("Alignment failed for scene %d (%s); using heuristic.", scene.index, exc)
            return self._fallback.align(scene)

        if not transcribed:
            return self._fallback.align(scene)

        timings = self._reconcile(scene, transcribed)
        if not timings:
            log.debug("Scene %d: transcript did not line up; using heuristic.", scene.index)
            return self._fallback.align(scene)
        return timings

    def _reconcile(self, scene: Scene, transcribed: list[dict]) -> list[WordTiming]:
        """Map transcript timings onto the script's own words.

        We caption the script text, not the transcript, so a mis-heard word does not
        show up on screen. Matching is positional after normalising both sides; if the
        counts diverge too far the caller falls back to the heuristic.
        """
        script_words = scene.narration.split()
        heard = [
            (str(item.get("word", "")).strip(), float(item.get("start", 0.0)),
             float(item.get("end", 0.0)))
            for item in transcribed
        ]
        heard = [item for item in heard if item[0]]
        if not heard:
            return []

        # More than a 25% length mismatch means the transcript is unreliable here.
        if abs(len(heard) - len(script_words)) > max(2, len(script_words) * 0.25):
            return []

        duration = scene.duration_ms or (heard[-1][2] * 1000.0)
        timings: list[WordTiming] = []
        for index, word in enumerate(script_words):
            if index < len(heard):
                _, start_s, end_s = heard[index]
                start_ms, end_ms = start_s * 1000.0, end_s * 1000.0
            else:
                # Ran off the end of the transcript: hold the tail evenly.
                start_ms = timings[-1].end_ms if timings else 0.0
                remaining = max(1, len(script_words) - index)
                end_ms = start_ms + max(0.0, duration - start_ms) / remaining

            start_ms = max(0.0, min(start_ms, duration))
            end_ms = max(start_ms, min(end_ms, duration))
            timings.append(WordTiming(word=word, start_ms=start_ms, end_ms=end_ms))

        return timings


def _weight(word: str) -> float:
    """Relative time cost of speaking ``word``."""
    core = re.sub(r"[^\w]", "", word)
    weight = max(_MIN_WEIGHT, float(len(core)))
    # Digits are read as full words ("2024" -> "twenty twenty-four").
    if any(ch.isdigit() for ch in core):
        weight += len(core) * 1.5
    for punct, extra in _PAUSE_WEIGHT.items():
        if word.endswith(punct):
            weight += extra
            break
    return weight


def build_aligner(strategy: str, provider: LLMProvider | None = None) -> Aligner:
    """Factory used by the pipeline."""
    if strategy == "whisper" and provider is not None and not provider.offline:
        return WhisperAligner(provider)
    if strategy == "whisper":
        log.info("Whisper alignment needs an online LLM provider; using the heuristic aligner.")
    return HeuristicAligner()
