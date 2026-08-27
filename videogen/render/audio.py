"""Narration assembly and background-music mixing.

The original mixed music by subtracting a flat number of decibels from the whole track
(``audio - 10 * log10(1/strength)``, which divides by zero at strength 0). A fixed
attenuation is always wrong somewhere: quiet enough to stay under the voice makes the
music inaudible in the gaps, loud enough to hear buries the narration.

This uses sidechain compression instead - the music ducks automatically whenever the
narrator speaks and returns to full level between lines, which is how broadcast does it.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..errors import RenderError
from ..logging import get_logger
from ..models import VideoSpec
from . import ffmpeg

log = get_logger("render.audio")

#: Gap inserted between scenes so lines do not run together.
SCENE_GAP_MS = 120.0


def concat_narration(
    audio_paths: list[Path],
    output_path: Path,
    *,
    gap_ms: float = SCENE_GAP_MS,
    settings: Settings | None = None,
) -> Path:
    """Join per-scene narration clips into one track, with a short gap between each."""
    if not audio_paths:
        raise RenderError("No narration audio to assemble.")

    args: list[str] = []
    for path in audio_paths:
        args += ["-i", str(path)]

    parts: list[str] = []
    labels: list[str] = []
    # Resample everything to a common format first; concat requires matching layouts.
    for index in range(len(audio_paths)):
        parts.append(f"[{index}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[a{index}]")
        labels.append(f"[a{index}]")

    if gap_ms > 0 and len(audio_paths) > 1:
        parts.append(f"aevalsrc=0:d={gap_ms / 1000:.3f}:s=44100:c=stereo[gap]")
        # asplit so the same silence source can be reused between every pair.
        gaps_needed = len(audio_paths) - 1
        parts.append(f"[gap]asplit={gaps_needed}" + "".join(f"[g{i}]" for i in range(gaps_needed)))
        interleaved: list[str] = []
        for index, label in enumerate(labels):
            interleaved.append(label)
            if index < gaps_needed:
                interleaved.append(f"[g{index}]")
        labels = interleaved

    parts.append("".join(labels) + f"concat=n={len(labels)}:v=0:a=1[out]")

    args += [
        "-filter_complex", ";".join(parts),
        "-map", "[out]",
        "-c:a", "libmp3lame", "-q:a", "2",
        str(output_path),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.run(args, settings=settings)
    log.debug("Assembled narration -> %s", output_path.name)
    return output_path


def mix_with_music(
    narration_path: Path,
    music_path: Path,
    output_path: Path,
    spec: VideoSpec,
    *,
    settings: Settings | None = None,
) -> Path:
    """Lay ``music_path`` under ``narration_path`` with automatic ducking.

    The music is looped to cover the narration, attenuated to ``spec.music_gain_db``,
    faded at both ends, then compressed against the narration as the sidechain so it
    drops by roughly ``spec.music_duck_db`` while the voice is present.
    """
    narration_ms = ffmpeg.duration_ms(narration_path, settings)
    duration_s = narration_ms / 1000.0
    fade_s = min(spec.music_fade_ms / 1000.0, max(0.0, duration_s / 3))

    # sidechaincompress expresses depth as a ratio; derive one from the requested dip.
    ratio = max(1.5, min(20.0, 10 ** (abs(spec.music_duck_db) / 20)))

    filters = [
        # Narration: normalise format, then split - one copy is mixed, one drives the
        # compressor's sidechain.
        "[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[nar]",
        "[nar]asplit=2[nar_mix][nar_key]",
        # Music: loop is handled by -stream_loop, so here just trim, level and fade.
        "[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
        f"atrim=0:{duration_s:.3f},asetpts=N/SR/TB,"
        f"volume={spec.music_gain_db:.2f}dB,"
        f"afade=t=in:st=0:d={fade_s:.3f},"
        f"afade=t=out:st={max(0.0, duration_s - fade_s):.3f}:d={fade_s:.3f}[music]",
        # Duck the music under the voice.
        f"[music][nar_key]sidechaincompress=threshold=0.03:ratio={ratio:.2f}:"
        "attack=20:release=350:makeup=1[ducked]",
        # normalize=0 keeps the narration at full level instead of halving it.
        "[nar_mix][ducked]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[out]",
    ]

    args = [
        "-i", str(narration_path),
        "-stream_loop", "-1", "-i", str(music_path),
        "-filter_complex", ";".join(filters),
        "-map", "[out]",
        "-t", f"{duration_s:.3f}",
        "-c:a", "libmp3lame", "-q:a", "2",
        str(output_path),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.run(args, settings=settings)
    log.info(
        "Mixed music under narration (%.1fs, %.0f dB bed, %.0f dB duck)",
        duration_s,
        spec.music_gain_db,
        spec.music_duck_db,
    )
    return output_path


def build_soundtrack(
    scene_audio: list[Path],
    output_dir: Path,
    spec: VideoSpec,
    *,
    music_path: Path | None = None,
    settings: Settings | None = None,
) -> tuple[Path, float]:
    """Produce the final audio track and report its duration.

    Returns ``(path, duration_ms)``. The duration is what the video track is rendered
    against, so picture and sound stay locked.
    """
    narration = concat_narration(
        scene_audio, output_dir / "narration.mp3", settings=settings
    )

    if music_path is not None and spec.music_enabled:
        if not music_path.exists():
            log.warning("Music file %s is missing; continuing without it.", music_path)
        else:
            mixed = mix_with_music(
                narration, music_path, output_dir / "soundtrack.mp3", spec, settings=settings
            )
            return mixed, ffmpeg.duration_ms(mixed, settings)

    return narration, ffmpeg.duration_ms(narration, settings)
