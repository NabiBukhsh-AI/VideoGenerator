"""Thin, typed wrapper around the ffmpeg/ffprobe binaries.

The original code wrote an XVID .avi with ``cv2.VideoWriter``, then shelled out to
ffmpeg to attach audio. That produced a large, dated-codec file no platform wants, and
it round-tripped every frame through an intermediate on disk.

Here frames are piped straight into ffmpeg as raw RGB and encoded once to H.264/MP4 with
``yuv420p``, which is what YouTube, TikTok and Instagram actually accept. Nothing is
written twice.
"""

from __future__ import annotations

import contextlib
import functools
import json
import shutil
import subprocess
import wave
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import IO

from ..config import Settings
from ..errors import DependencyError, FFmpegError
from ..logging import get_logger

log = get_logger("render.ffmpeg")

_INSTALL_HINT = (
    "Install ffmpeg and make sure it is on PATH.\n"
    "  Windows: winget install Gyan.FFmpeg   (or: choco install ffmpeg)\n"
    "  macOS:   brew install ffmpeg\n"
    "  Linux:   sudo apt install ffmpeg\n"
    "Then restart your shell so PATH is picked up."
)


@functools.lru_cache(maxsize=8)
def _resolve(binary: str) -> str | None:
    return shutil.which(binary)


def find_ffmpeg(settings: Settings | None = None) -> str | None:
    name = settings.ffmpeg_binary if settings else "ffmpeg"
    return _resolve(name)


def find_ffprobe(settings: Settings | None = None) -> str | None:
    name = settings.ffprobe_binary if settings else "ffprobe"
    return _resolve(name)


def require_ffmpeg(settings: Settings | None = None) -> str:
    path = find_ffmpeg(settings)
    if path is None:
        raise DependencyError("ffmpeg was not found on PATH.", hint=_INSTALL_HINT)
    return path


def available(settings: Settings | None = None) -> bool:
    """True when a video can actually be encoded on this machine."""
    return find_ffmpeg(settings) is not None


def version(settings: Settings | None = None) -> str:
    path = find_ffmpeg(settings)
    if path is None:
        return "not found"
    try:
        out = subprocess.run(
            [path, "-version"], capture_output=True, text=True, timeout=15, check=False
        )
        return out.stdout.splitlines()[0] if out.stdout else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run(args: Sequence[str], *, settings: Settings | None = None, timeout: float = 3600) -> str:
    """Run ffmpeg with ``args`` (binary prepended). Returns stderr on success."""
    command = [require_ffmpeg(settings), "-hide_banner", "-loglevel", "error", "-y", *args]
    log.debug("ffmpeg %s", " ".join(args))
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"ffmpeg timed out after {timeout}s", command=command) from exc
    except OSError as exc:
        raise FFmpegError(f"Could not run ffmpeg: {exc}", command=command) from exc

    if result.returncode != 0:
        raise FFmpegError(
            f"ffmpeg failed (exit {result.returncode}): {_tail(result.stderr)}",
            command=command,
            stderr=result.stderr,
        )
    return result.stderr


def probe(path: Path, settings: Settings | None = None) -> dict:
    """Return ffprobe's JSON for ``path``."""
    binary = find_ffprobe(settings)
    if binary is None:
        raise DependencyError("ffprobe was not found on PATH.", hint=_INSTALL_HINT)

    command = [
        binary, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FFmpegError(f"ffprobe failed on {path.name}: {exc}", command=command) from exc

    if result.returncode != 0:
        raise FFmpegError(
            f"ffprobe failed on {path.name}: {_tail(result.stderr)}",
            command=command,
            stderr=result.stderr,
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"ffprobe returned unparseable JSON for {path.name}") from exc


def duration_ms(path: Path, settings: Settings | None = None) -> float:
    """Duration of a media file in milliseconds.

    WAV is measured with the standard library so offline runs and CI never need ffmpeg.
    """
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
                if rate:
                    return frames / rate * 1000.0
        except (wave.Error, OSError) as exc:
            log.debug("wave module could not read %s (%s); falling back to ffprobe", path.name, exc)

    info = probe(path, settings)
    value = info.get("format", {}).get("duration")
    if value is None:
        for stream in info.get("streams", []):
            if stream.get("duration"):
                value = stream["duration"]
                break
    if value is None:
        raise FFmpegError(f"Could not determine the duration of {path.name}")
    return float(value) * 1000.0


@contextmanager
def frame_writer(
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    crf: int = 20,
    preset: str = "medium",
    settings: Settings | None = None,
) -> Iterator[IO[bytes]]:
    """Open an ffmpeg process that accepts raw RGB frames on stdin.

    Yields the stdin pipe. Write exactly ``width * height * 3`` bytes per frame.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        require_ffmpeg(settings),
        "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",          # required for playback on phones and Safari
        "-movflags", "+faststart",      # metadata first, so the file streams
        str(output_path),
    ]
    log.debug("Encoding %dx%d @ %dfps -> %s", width, height, fps, output_path.name)

    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    assert process.stdin is not None
    try:
        yield process.stdin
    except BaseException:
        process.kill()
        process.wait(timeout=10)
        raise
    finally:
        if not process.stdin.closed:
            # ffmpeg may already have exited; the returncode check below reports why.
            with contextlib.suppress(BrokenPipeError):
                process.stdin.close()

    stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
    if process.wait() != 0:
        raise FFmpegError(
            f"Encoding failed (exit {process.returncode}): {_tail(stderr)}",
            command=command,
            stderr=stderr,
        )


def mux(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    audio_bitrate: str = "192k",
    shortest: bool = True,
    settings: Settings | None = None,
) -> Path:
    """Combine an encoded video track with an audio track without re-encoding video."""
    args = [
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",
    ]
    if shortest:
        args.append("-shortest")
    args.append(str(output_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(args, settings=settings)
    return output_path


def extract_frame(video_path: Path, output_path: Path, *, at_ms: float = 0,
                  settings: Settings | None = None) -> Path:
    """Grab a single frame as a JPEG - used for the thumbnail."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "-ss", f"{at_ms / 1000:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(output_path),
        ],
        settings=settings,
    )
    return output_path


def _tail(text: str, lines: int = 6) -> str:
    """Last few lines of stderr - ffmpeg puts the actual error at the end."""
    if not text:
        return "(no output)"
    return " | ".join(text.strip().splitlines()[-lines:])
