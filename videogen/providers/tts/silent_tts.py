"""Offline TTS stub.

Writes a real, valid WAV file of silence whose length matches how long the sentence
would take to read aloud. That matters more than it sounds: the renderer derives every
scene's on-screen duration from audio length, so a stub that returns plausible
durations exercises the whole timing path.

WAV is written with the standard library, so dry runs and CI need no ffmpeg and no
audio dependencies at all.
"""

from __future__ import annotations

import re
import wave
from pathlib import Path

from ..base import TTSProvider

# Speaking rate for an energetic short-form narrator.
_WORDS_PER_MINUTE = 155.0
_MIN_DURATION_MS = 900.0
# Punctuation makes a real narrator pause; mirror that so pacing feels right.
_PAUSE_MS = {",": 180.0, ";": 220.0, ":": 220.0, ".": 320.0, "!": 320.0, "?": 340.0}

_SAMPLE_RATE = 44100
_CHANNELS = 1
_SAMPLE_WIDTH = 2  # 16-bit


class SilentTTS(TTSProvider):
    name = "silent"
    offline = True
    audio_format = "wav"

    def synthesize(self, *, text: str, output_path: Path, voice: str | None = None) -> Path:
        duration_ms = estimate_speech_ms(text)
        output_path = output_path.with_suffix(".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        frames = int(_SAMPLE_RATE * duration_ms / 1000.0)
        with wave.open(str(output_path), "wb") as handle:
            handle.setnchannels(_CHANNELS)
            handle.setsampwidth(_SAMPLE_WIDTH)
            handle.setframerate(_SAMPLE_RATE)
            handle.writeframes(b"\x00" * (frames * _CHANNELS * _SAMPLE_WIDTH))
        return output_path


def estimate_speech_ms(text: str) -> float:
    """Estimate how long ``text`` takes to narrate, in milliseconds.

    Also used by the CLI to preview a script's runtime before any audio exists.
    """
    words = len(text.split())
    base = words / _WORDS_PER_MINUTE * 60_000
    pauses = sum(_PAUSE_MS.get(ch, 0.0) for ch in re.findall(r"[,;:.!?]", text))
    return max(_MIN_DURATION_MS, base + pauses)
