"""Animated caption rendering.

What was wrong with the original `write_text`: it centred the line by measuring the
whole string, then drew word by word advancing the cursor by ``word_width + 10``. The
sum of those advances does not equal the measured width, so every line drifted right
and long lines ran off the frame. It also never wrapped by pixel width, only by a fixed
word count, and OpenCV's Hershey font cannot draw non-ASCII at all.

This module lays out a line by measuring the exact space width in the actual font,
wraps on both word count and pixel width, and renders with PIL so any script the font
supports works.

Two things keep it fast enough for 30fps: overlays are cached per (line, highlighted
word) because consecutive frames reuse them, and each overlay is cropped to its own
bounding box so compositing touches a caption-sized region rather than the whole frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from ..models import Scene, VideoSpec, WordTiming
from .fonts import FontLike, font_metrics, load_font, measure


@dataclass
class CaptionLine:
    """One on-screen line with the timing of each word in it."""

    words: list[str]
    timings: list[WordTiming]

    @property
    def text(self) -> str:
        return " ".join(self.words)

    @property
    def start_ms(self) -> float:
        return self.timings[0].start_ms if self.timings else 0.0

    @property
    def end_ms(self) -> float:
        return self.timings[-1].end_ms if self.timings else 0.0

    def active_word(self, t_ms: float) -> int:
        """Index of the word being spoken at ``t_ms``, clamped to the line."""
        for index, timing in enumerate(self.timings):
            if t_ms < timing.end_ms:
                return index
        return max(0, len(self.words) - 1)


@dataclass
class CaptionPlan:
    """All caption lines for one scene, in order."""

    lines: list[CaptionLine] = field(default_factory=list)

    def line_at(self, t_ms: float) -> CaptionLine | None:
        for line in self.lines:
            if line.start_ms <= t_ms < line.end_ms:
                return line
        # Past the last word (trailing silence): hold the final line rather than
        # flashing to nothing.
        if self.lines and t_ms >= self.lines[-1].end_ms:
            return self.lines[-1]
        return None


@dataclass(frozen=True)
class Overlay:
    """A caption image cropped to its bounding box, ready to blend onto a frame."""

    rgb: np.ndarray  # (h, w, 3) uint8
    alpha: np.ndarray  # (h, w, 1) uint16, 0-256
    x: int
    y: int

    def composite(self, frame: np.ndarray) -> np.ndarray:
        """Return ``frame`` with this overlay blended in. The input is not mutated."""
        height, width = self.rgb.shape[:2]
        if height == 0 or width == 0:
            return frame

        out = frame.copy()
        region = out[self.y : self.y + height, self.x : self.x + width]
        # Integer alpha blend: dst = (dst*(256-a) + src*a) >> 8
        blended = (
            region.astype(np.uint16) * (256 - self.alpha) + self.rgb.astype(np.uint16) * self.alpha
        ) >> 8
        out[self.y : self.y + height, self.x : self.x + width] = blended.astype(np.uint8)
        return out


class CaptionRenderer:
    """Builds caption plans and renders them to compositable overlays."""

    def __init__(self, spec: VideoSpec) -> None:
        self.spec = spec
        self.width, self.height = spec.size
        self.font_size = spec.resolved_font_size()
        self.stroke_width = spec.resolved_stroke_width()
        self.font: FontLike = load_font(self.font_size)
        self.max_line_width = self.width * 0.86

        _, self.line_height = font_metrics(self.font)
        self.space_width = measure(" ", self.font)

        self.color = _parse_color(spec.caption_color)
        self.highlight = _parse_color(spec.caption_highlight_color)
        self.stroke_color = _parse_color(spec.caption_stroke_color)

        # Bounded so a long run cannot grow memory without limit.
        self._cache: dict[tuple[str, int], Overlay] = {}
        self._cache_limit = 512

    # -- planning --------------------------------------------------------------
    def plan(self, scene: Scene) -> CaptionPlan:
        """Group a scene's word timings into displayable lines."""
        timings = scene.word_timings
        if not timings:
            return CaptionPlan()

        lines: list[CaptionLine] = []
        words: list[str] = []
        spans: list[WordTiming] = []

        for timing in timings:
            candidate = [*words, timing.word]
            too_many = len(candidate) > self.spec.caption_words_per_line
            too_wide = self._line_width(candidate) > self.max_line_width

            if words and (too_many or too_wide):
                lines.append(CaptionLine(words=words, timings=spans))
                words, spans = [], []

            words.append(timing.word)
            spans.append(timing)

        if words:
            lines.append(CaptionLine(words=words, timings=spans))

        return CaptionPlan(lines=lines)

    def _line_width(self, words: list[str]) -> float:
        if not words:
            return 0.0
        return sum(measure(word, self.font) for word in words) + self.space_width * (
            len(words) - 1
        )

    # -- rendering -------------------------------------------------------------
    def overlay(self, line: CaptionLine, highlighted: int) -> Overlay:
        """Overlay for ``line`` with word ``highlighted`` emphasised."""
        key = (line.text, highlighted)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        rows = self._wrap_rows(line.words)
        total_height = self.line_height * len(rows)
        top_y = self.height * (1 - self.spec.caption_bottom_margin) - total_height / 2

        for row_number, row in enumerate(rows):
            row_words = [line.words[i] for i in row]
            x = (self.width - self._line_width(row_words)) / 2
            y = top_y + row_number * self.line_height

            for word_index, word in zip(row, row_words, strict=True):
                draw.text(
                    (x, y),
                    word,
                    font=self.font,
                    fill=self.highlight if word_index == highlighted else self.color,
                    stroke_width=self.stroke_width,
                    stroke_fill=self.stroke_color,
                )
                x += measure(word, self.font) + self.space_width

        result = self._to_overlay(canvas)

        if len(self._cache) >= self._cache_limit:
            self._cache.clear()
        self._cache[key] = result
        return result

    def _to_overlay(self, canvas: Image.Image) -> Overlay:
        """Crop to the drawn area and split into RGB + alpha arrays."""
        box = canvas.getbbox()
        if box is None:
            empty = np.zeros((0, 0, 3), dtype=np.uint8)
            return Overlay(rgb=empty, alpha=np.zeros((0, 0, 1), dtype=np.uint16), x=0, y=0)

        left, top, right, bottom = box
        # Clamp to the canvas: stroke can push the bbox past the edge.
        left, top = max(0, left), max(0, top)
        right, bottom = min(self.width, right), min(self.height, bottom)

        cropped = np.asarray(canvas.crop((left, top, right, bottom)), dtype=np.uint8)
        alpha = cropped[:, :, 3:4].astype(np.uint16)
        # Map 0-255 to 0-256 so the >>8 blend reaches full opacity.
        alpha = alpha + (alpha >> 7)
        return Overlay(rgb=cropped[:, :, :3].copy(), alpha=alpha, x=left, y=top)

    def _wrap_rows(self, words: list[str]) -> list[list[int]]:
        """Split word indices into rows that fit the line box."""
        rows: list[list[int]] = []
        current: list[int] = []
        for index in range(len(words)):
            candidate = [*current, index]
            if current and self._line_width([words[i] for i in candidate]) > self.max_line_width:
                rows.append(current)
                current = [index]
            else:
                current = candidate
        if current:
            rows.append(current)
        return rows or [[]]

    def overlay_at(self, plan: CaptionPlan, t_ms: float) -> Overlay | None:
        """Overlay for scene-relative time ``t_ms``, or None when nothing shows."""
        line = plan.line_at(t_ms)
        if line is None or not line.words:
            return None
        return self.overlay(line, line.active_word(t_ms))


@lru_cache(maxsize=64)
def _parse_color(value: str) -> tuple[int, int, int, int]:
    """Parse ``#RGB``, ``#RRGGBB``, ``#RRGGBBAA`` or a PIL colour name to RGBA."""
    text = value.strip()
    if text.startswith("#"):
        hexed = text[1:]
        if len(hexed) == 3:
            hexed = "".join(ch * 2 for ch in hexed)
        if len(hexed) == 6:
            hexed += "ff"
        if len(hexed) == 8:
            try:
                return (
                    int(hexed[0:2], 16),
                    int(hexed[2:4], 16),
                    int(hexed[4:6], 16),
                    int(hexed[6:8], 16),
                )
            except ValueError:
                pass

    try:
        from PIL import ImageColor

        rgb = ImageColor.getrgb(text)
        return (*rgb, 255) if len(rgb) == 3 else rgb  # type: ignore[return-value]
    except (ImportError, ValueError):
        return (255, 255, 255, 255)
