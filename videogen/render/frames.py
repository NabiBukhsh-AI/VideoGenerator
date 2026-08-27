"""Frame generation: Ken Burns motion, transitions and the scene timeline.

Three concrete fixes over the original renderer:

* **Framing.** It scaled each image to fit inside the canvas and pasted it at the
  top-left of a black frame, so vertical videos had black bars and off-centre
  subjects. Images are now cover-cropped and centred, filling the frame exactly.
* **Motion.** Every shot was a static still. Each scene now gets a slow zoom or pan,
  which is what stops a slideshow from reading as a slideshow.
* **Timing.** The old code subtracted the fade length from some scene durations and not
  others, so picture and narration drifted apart over the video. The timeline here
  places scenes end to end at their exact audio durations, and the crossfade happens
  inside the outgoing scene's own window, so total picture length always equals total
  audio length.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..errors import RenderError
from ..logging import get_logger
from ..models import KenBurnsKind, Scene, TransitionKind, VideoSpec
from .captions import CaptionPlan, CaptionRenderer

log = get_logger("render.frames")

# Rotating pattern used when ken_burns is AUTO. Alternating direction keeps a
# multi-scene video from feeling mechanical.
_AUTO_CYCLE = (
    KenBurnsKind.ZOOM_IN,
    KenBurnsKind.PAN_RIGHT,
    KenBurnsKind.ZOOM_OUT,
    KenBurnsKind.PAN_LEFT,
)


def load_cover(path: Path, size: tuple[int, int], *, headroom: float = 0.0) -> Image.Image:
    """Load an image cropped to ``size``'s aspect and scaled to cover it.

    ``headroom`` oversizes the result (e.g. 0.12 for 12% extra) so Ken Burns has room
    to move without ever exposing an edge.
    """
    try:
        source = Image.open(path)
        source.load()
    except (OSError, ValueError) as exc:
        raise RenderError(f"Could not read image {path.name}: {exc}") from exc

    if source.mode != "RGB":
        source = source.convert("RGB")

    target_w = round(size[0] * (1 + headroom))
    target_h = round(size[1] * (1 + headroom))
    target_aspect = target_w / target_h
    source_aspect = source.width / source.height

    # Crop the longer axis so the visible area matches the canvas aspect exactly.
    if source_aspect > target_aspect:
        crop_w = round(source.height * target_aspect)
        left = (source.width - crop_w) // 2
        source = source.crop((left, 0, left + crop_w, source.height))
    elif source_aspect < target_aspect:
        crop_h = round(source.width / target_aspect)
        top = (source.height - crop_h) // 2
        source = source.crop((0, top, source.width, top + crop_h))

    return source.resize((target_w, target_h), Image.LANCZOS)


@dataclass
class SceneClip:
    """A scene's visual: a prepared image plus the motion applied to it."""

    image: Image.Image
    duration_ms: float
    motion: KenBurnsKind
    intensity: float
    canvas: tuple[int, int]
    plan: CaptionPlan | None = None

    def __post_init__(self) -> None:
        self._static: np.ndarray | None = None

    def render(self, progress: float) -> np.ndarray:
        """Frame at ``progress`` in [0, 1] as an HxWx3 uint8 RGB array."""
        progress = min(1.0, max(0.0, progress))

        if self.motion is KenBurnsKind.NONE or self.intensity <= 0:
            if self._static is None:
                fitted = self.image if self.image.size == self.canvas else self.image.resize(
                    self.canvas, Image.LANCZOS
                )
                self._static = np.asarray(fitted, dtype=np.uint8)
            return self._static

        box = self._crop_box(progress)
        frame = self.image.resize(self.canvas, Image.LANCZOS, box=box)
        return np.asarray(frame, dtype=np.uint8)

    def _crop_box(self, progress: float) -> tuple[float, float, float, float]:
        """Crop rectangle in source pixels for this point in the move."""
        source_w, source_h = self.image.size
        width, height = self.canvas
        span_x = source_w - width
        span_y = source_h - height

        if self.motion is KenBurnsKind.ZOOM_IN:
            # Start at full extent, end tightly cropped in the centre.
            scale = 1.0 - self.intensity * progress
        elif self.motion is KenBurnsKind.ZOOM_OUT:
            scale = 1.0 - self.intensity * (1.0 - progress)
        else:
            scale = 1.0

        crop_w = min(source_w, width / max(scale, 0.05))
        crop_h = min(source_h, height / max(scale, 0.05))

        if self.motion is KenBurnsKind.PAN_LEFT:
            left = max(0.0, span_x) * (1.0 - progress)
            top = max(0.0, span_y) / 2
        elif self.motion is KenBurnsKind.PAN_RIGHT:
            left = max(0.0, span_x) * progress
            top = max(0.0, span_y) / 2
        else:
            left = (source_w - crop_w) / 2
            top = (source_h - crop_h) / 2

        left = min(max(0.0, left), max(0.0, source_w - crop_w))
        top = min(max(0.0, top), max(0.0, source_h - crop_h))
        return (left, top, left + crop_w, top + crop_h)


class Timeline:
    """Places scenes on a clock and produces every frame of the video."""

    def __init__(
        self,
        clips: list[SceneClip],
        spec: VideoSpec,
        captions: CaptionRenderer | None = None,
    ) -> None:
        if not clips:
            raise RenderError("Cannot build a timeline with no scenes.")
        self.clips = clips
        self.spec = spec
        self.captions = captions

        self.starts: list[float] = []
        cursor = 0.0
        for clip in clips:
            self.starts.append(cursor)
            cursor += clip.duration_ms
        self.total_ms = cursor

    @property
    def frame_count(self) -> int:
        return max(1, round(self.total_ms / 1000.0 * self.spec.fps))

    def _index_at(self, t_ms: float) -> int:
        # Linear scan is fine: scene counts are single digits.
        for index in range(len(self.clips) - 1, -1, -1):
            if t_ms >= self.starts[index]:
                return index
        return 0

    def frame_at(self, t_ms: float) -> np.ndarray:
        index = self._index_at(t_ms)
        clip = self.clips[index]
        local_ms = t_ms - self.starts[index]
        progress = local_ms / clip.duration_ms if clip.duration_ms > 0 else 0.0
        frame = clip.render(progress)

        frame = self._apply_transition(frame, index, local_ms)

        if self.captions is not None and clip.plan is not None:
            frame = self._apply_caption(frame, clip, local_ms)
        return frame

    def _apply_transition(self, frame: np.ndarray, index: int, local_ms: float) -> np.ndarray:
        """Blend into the next scene during the outgoing scene's tail."""
        spec = self.spec
        if spec.transition is TransitionKind.CUT or spec.transition_ms <= 0:
            return frame

        clip = self.clips[index]
        transition_ms = min(spec.transition_ms, clip.duration_ms)
        tail_start = clip.duration_ms - transition_ms
        if local_ms < tail_start:
            return frame

        alpha = (local_ms - tail_start) / transition_ms if transition_ms else 1.0
        alpha = min(1.0, max(0.0, alpha))

        is_last = index >= len(self.clips) - 1
        if is_last or spec.transition is TransitionKind.FADE_TO_BLACK:
            # Fade the final scene out; mid-video fade-to-black dips through black.
            if is_last:
                return _blend(frame, np.zeros_like(frame), alpha)
            incoming = self.clips[index + 1].render(0.0)
            if alpha < 0.5:
                return _blend(frame, np.zeros_like(frame), alpha * 2)
            return _blend(np.zeros_like(frame), incoming, (alpha - 0.5) * 2)

        return _blend(frame, self.clips[index + 1].render(0.0), alpha)

    def _apply_caption(self, frame: np.ndarray, clip: SceneClip, local_ms: float) -> np.ndarray:
        assert self.captions is not None and clip.plan is not None
        overlay = self.captions.overlay_at(clip.plan, local_ms)
        if overlay is None:
            return frame
        return overlay.composite(frame)

    def frames(self) -> Iterator[np.ndarray]:
        """Yield every frame in order."""
        step_ms = 1000.0 / self.spec.fps
        for number in range(self.frame_count):
            yield self.frame_at(number * step_ms)


def _blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Linear cross-dissolve from ``a`` to ``b``, done in uint16 to avoid overflow."""
    if alpha <= 0:
        return a
    if alpha >= 1:
        return b
    weight = int(alpha * 256)
    return ((a.astype(np.uint16) * (256 - weight) + b.astype(np.uint16) * weight) >> 8).astype(
        np.uint8
    )


def build_clips(
    scenes: list[Scene],
    spec: VideoSpec,
    captions: CaptionRenderer | None = None,
) -> list[SceneClip]:
    """Prepare a `SceneClip` per scene, resolving AUTO motion to a concrete move."""
    clips: list[SceneClip] = []
    headroom = spec.ken_burns_intensity if spec.ken_burns is not KenBurnsKind.NONE else 0.0

    for position, scene in enumerate(scenes):
        if scene.image_path is None:
            raise RenderError(f"Scene {scene.index} has no image.")
        if not scene.duration_ms:
            raise RenderError(f"Scene {scene.index} has no narration duration.")

        motion = spec.ken_burns
        if motion is KenBurnsKind.AUTO:
            motion = _AUTO_CYCLE[position % len(_AUTO_CYCLE)]

        clips.append(
            SceneClip(
                image=load_cover(scene.image_path, spec.size, headroom=headroom),
                duration_ms=scene.duration_ms,
                motion=motion,
                intensity=spec.ken_burns_intensity,
                canvas=spec.size,
                plan=captions.plan(scene) if captions is not None else None,
            )
        )

    log.debug("Prepared %d clips totalling %.1fs", len(clips),
              sum(c.duration_ms for c in clips) / 1000)
    return clips
