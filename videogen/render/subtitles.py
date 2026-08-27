"""Subtitle export (SRT and WebVTT).

New capability. Burned-in captions look good but are invisible to platforms; a sidecar
file gives you searchable, translatable, accessible subtitles for the same video.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..models import Scene


@dataclass
class Cue:
    """One subtitle entry on the absolute video clock."""

    index: int
    start_ms: float
    end_ms: float
    text: str


def build_cues(scenes: list[Scene], *, gap_ms: float = 0.0, max_chars: int = 84) -> list[Cue]:
    """Build cues from scene narration, offset onto the absolute timeline.

    One cue per scene where the line is short enough to read; longer narration is split
    at word boundaries using the per-word timings so each cue stays legible.
    """
    cues: list[Cue] = []
    cursor = 0.0

    for scene in scenes:
        duration = scene.duration_ms or 0.0
        if duration <= 0:
            continue

        for start_ms, end_ms, text in _split_scene(scene, duration, max_chars):
            cues.append(
                Cue(
                    index=len(cues) + 1,
                    start_ms=cursor + start_ms,
                    end_ms=cursor + end_ms,
                    text=text,
                )
            )
        cursor += duration + gap_ms

    return cues


def _split_scene(scene: Scene, duration: float, max_chars: int) -> list[tuple[float, float, str]]:
    text = scene.narration.strip()
    if len(text) <= max_chars or not scene.word_timings:
        return [(0.0, duration, text)]

    chunks: list[tuple[float, float, str]] = []
    words: list[str] = []
    start = scene.word_timings[0].start_ms

    for timing in scene.word_timings:
        candidate = " ".join([*words, timing.word])
        if words and len(candidate) > max_chars:
            chunks.append((start, timing.start_ms, " ".join(words)))
            words, start = [timing.word], timing.start_ms
        else:
            words.append(timing.word)

    if words:
        chunks.append((start, duration, " ".join(words)))
    return chunks


def write_srt(cues: list[Cue], output_path: Path) -> Path:
    """Write cues in SubRip format."""
    blocks = [
        f"{cue.index}\n"
        f"{_timestamp(cue.start_ms, ',')} --> {_timestamp(cue.end_ms, ',')}\n"
        f"{cue.text}"
        for cue in cues
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return output_path


def write_vtt(cues: list[Cue], output_path: Path) -> Path:
    """Write cues in WebVTT format."""
    blocks = ["WEBVTT", ""]
    for cue in cues:
        blocks.append(f"{_timestamp(cue.start_ms, '.')} --> {_timestamp(cue.end_ms, '.')}")
        blocks.append(cue.text)
        blocks.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(blocks), encoding="utf-8")
    return output_path


def _timestamp(milliseconds: float, decimal: str) -> str:
    """Format as ``HH:MM:SS,mmm`` (SRT) or ``HH:MM:SS.mmm`` (VTT)."""
    total_ms = max(0, round(milliseconds))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{decimal}{ms:03d}"
