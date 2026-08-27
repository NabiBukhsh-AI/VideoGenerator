"""Offline image provider.

Renders a deterministic gradient card carrying the prompt text. Same prompt always
yields the same image, which keeps dry runs and snapshot tests reproducible.
"""

from __future__ import annotations

import colorsys
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from ...render.fonts import load_font, wrap_text
from ..base import ImageProvider


class PlaceholderImage(ImageProvider):
    name = "placeholder"
    offline = True

    def generate(self, *, prompt: str, output_path: Path, size: str = "1024x1536") -> Path:
        width, height = _parse_size(size)
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()

        # Two hues a fixed distance apart give a readable, varied gradient.
        hue = digest[0] / 255.0
        top = _hsv_to_rgb(hue, 0.55, 0.42)
        bottom = _hsv_to_rgb((hue + 0.12) % 1.0, 0.65, 0.16)

        image = Image.new("RGB", (width, height), top)
        draw = ImageDraw.Draw(image)
        for y in range(height):
            blend = y / max(1, height - 1)
            draw.line(
                [(0, y), (width, y)],
                fill=tuple(round(top[i] + (bottom[i] - top[i]) * blend) for i in range(3)),
            )

        _draw_caption(draw, prompt, width, height)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")
        return output_path


def _draw_caption(draw: ImageDraw.ImageDraw, prompt: str, width: int, height: int) -> None:
    font_size = max(18, width // 26)
    font = load_font(font_size)
    lines = wrap_text(prompt[:220], font, int(width * 0.82), draw)[:6]

    line_height = font_size * 1.35
    total = line_height * len(lines)
    y = (height - total) / 2

    for line in lines:
        text_width = draw.textlength(line, font=font)
        draw.text(
            ((width - text_width) / 2, y),
            line,
            font=font,
            fill=(255, 255, 255),
            stroke_width=max(1, font_size // 14),
            stroke_fill=(0, 0, 0),
        )
        y += line_height


def _parse_size(size: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in size.lower().split("x", 1))
        return max(16, width), max(16, height)
    except (ValueError, AttributeError):
        return 1024, 1536


def _hsv_to_rgb(hue: float, saturation: float, value: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return round(r * 255), round(g * 255), round(b * 255)
