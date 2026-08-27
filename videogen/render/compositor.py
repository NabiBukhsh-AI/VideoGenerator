"""Render orchestration: frames in, finished MP4 out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Settings
from ..errors import RenderError
from ..logging import ProgressReporter, get_logger
from ..models import Script, VideoSpec
from ..utils import Timer, format_duration
from . import audio as audio_mod
from . import ffmpeg, subtitles
from .captions import CaptionRenderer
from .frames import Timeline, build_clips

log = get_logger("render")


@dataclass
class CompositionResult:
    video_path: Path
    duration_ms: float
    subtitle_path: Path | None = None
    thumbnail_path: Path | None = None


class Compositor:
    """Turns a fully populated `Script` into a video file."""

    def __init__(self, spec: VideoSpec, settings: Settings) -> None:
        self.spec = spec
        self.settings = settings

    def render(
        self,
        script: Script,
        work_dir: Path,
        output_path: Path,
        *,
        music_path: Path | None = None,
        progress: ProgressReporter | None = None,
        write_subtitles: bool = True,
        write_thumbnail: bool = True,
    ) -> CompositionResult:
        progress = progress or ProgressReporter(log)
        work_dir.mkdir(parents=True, exist_ok=True)

        missing = [s.index for s in script.scenes if not s.is_rendered]
        if missing:
            raise RenderError(f"Scenes {missing} are missing an image, audio or duration.")

        # 1. Soundtrack first: its length is the contract the picture is cut to.
        progress.stage("Assembling audio")
        with Timer("audio", log):
            soundtrack, audio_ms = audio_mod.build_soundtrack(
                [s.audio_path for s in script.scenes if s.audio_path],
                work_dir,
                self.spec,
                music_path=music_path,
                settings=self.settings,
            )

        # 2. Build the visual timeline, padding each scene by the same inter-scene gap
        #    the audio concatenation inserted so picture and sound stay locked.
        captions = CaptionRenderer(self.spec) if self.spec.captions_enabled else None
        clips = build_clips(script.scenes, self.spec, captions)
        gap_ms = audio_mod.SCENE_GAP_MS if len(clips) > 1 else 0.0
        for clip in clips[:-1]:
            clip.duration_ms += gap_ms

        timeline = Timeline(clips, self.spec, captions)
        drift_ms = abs(timeline.total_ms - audio_ms)
        if drift_ms > 250:
            log.warning(
                "Picture (%s) and audio (%s) differ by %.0fms; ffmpeg will trim to the shorter.",
                format_duration(timeline.total_ms),
                format_duration(audio_ms),
                drift_ms,
            )

        # 3. Encode the picture.
        silent_video = work_dir / "video_track.mp4"
        total_frames = timeline.frame_count
        progress.stage("Rendering video", f"{total_frames} frames at {self.spec.fps}fps")

        with Timer("frames", log):
            self._encode(timeline, silent_video, total_frames, progress)

        # 4. Mux.
        progress.stage("Muxing audio and video")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg.mux(
            silent_video,
            soundtrack,
            output_path,
            audio_bitrate=self.spec.audio_bitrate,
            settings=self.settings,
        )

        duration_ms = min(timeline.total_ms, audio_ms)
        result = CompositionResult(video_path=output_path, duration_ms=duration_ms)

        # 5. Sidecars.
        if write_subtitles:
            cues = subtitles.build_cues(script.scenes, gap_ms=gap_ms)
            result.subtitle_path = subtitles.write_srt(
                cues, output_path.with_suffix(".srt")
            )
            subtitles.write_vtt(cues, output_path.with_suffix(".vtt"))

        if write_thumbnail:
            try:
                result.thumbnail_path = ffmpeg.extract_frame(
                    output_path,
                    output_path.with_suffix(".jpg"),
                    at_ms=min(1200, duration_ms / 4),
                    settings=self.settings,
                )
            except Exception as exc:
                log.warning("Could not extract a thumbnail: %s", exc)

        silent_video.unlink(missing_ok=True)
        log.info("Rendered %s (%s)", output_path.name, format_duration(duration_ms))
        return result

    def _encode(
        self,
        timeline: Timeline,
        output_path: Path,
        total_frames: int,
        progress: ProgressReporter,
    ) -> None:
        """Stream frames into ffmpeg, reporting progress as we go."""
        # Report roughly 100 times over the run rather than on every frame.
        report_every = max(1, total_frames // 100)

        with ffmpeg.frame_writer(
            output_path,
            width=self.spec.width,
            height=self.spec.height,
            fps=self.spec.fps,
            crf=self.spec.crf,
            preset=self.spec.preset,
            settings=self.settings,
        ) as pipe:
            for number, frame in enumerate(timeline.frames()):
                try:
                    pipe.write(frame.tobytes())
                except BrokenPipeError as exc:
                    # ffmpeg died mid-stream; frame_writer surfaces its stderr on exit.
                    raise RenderError(
                        "ffmpeg closed the stream while frames were still being written."
                    ) from exc

                if number % report_every == 0:
                    progress.update(number / total_frames, f"frame {number}/{total_frames}")

        progress.update(1.0, f"{total_frames} frames")
