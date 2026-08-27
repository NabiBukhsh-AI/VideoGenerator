"""Font resolution and text measurement.

The original renderer used OpenCV's Hershey vector fonts, which cannot draw anything
outside ASCII and look like 1980s CAD output. Captions here go through PIL with a real
TrueType face, so non-Latin scripts render and the result looks like a modern short.

Finding a usable TTF on an arbitrary machine is the fiddly part: this walks the
platform's font directories, then falls back to whatever matplotlib shipped, and only
then to PIL's bitmap default.
"""

from __future__ import annotations

import functools
from pathlib import Path

from PIL import ImageDraw, ImageFont

from ..logging import get_logger

log = get_logger("render.fonts")

FontLike = ImageFont.FreeTypeFont | ImageFont.ImageFont

# Ordered by preference: a heavy weight reads far better as a burned-in caption.
_CANDIDATES: tuple[str, ...] = (
    # Windows
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\Arial.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
)

_ASSET_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts"

_override: Path | None = None


def set_font_override(path: Path | str | None) -> None:
    """Pin a specific TTF for the rest of the process (``--font`` / settings)."""
    global _override
    _override = Path(path) if path else None
    load_font.cache_clear()


def _discover() -> Path | None:
    if _override and _override.exists():
        return _override

    if _ASSET_DIR.is_dir():
        for candidate in sorted(_ASSET_DIR.glob("*.tt[fc]")):
            return candidate

    for candidate in _CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path

    # matplotlib bundles DejaVu and is very often already installed.
    try:
        import matplotlib

        bundled = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        if bundled.exists():
            return bundled
    except Exception:
        pass

    return None


@functools.lru_cache(maxsize=32)
def load_font(size: int) -> FontLike:
    """Return a font at ``size``, cached per size.

    Falls back to PIL's bitmap font rather than raising: a run that produces ugly
    captions is still more useful than one that dies at the render step.
    """
    path = _discover()
    if path is not None:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError as exc:
            log.warning("Could not load font %s (%s); falling back to PIL default", path, exc)

    log.warning(
        "No TrueType font found - captions will use PIL's bitmap font and ignore size. "
        "Set VIDEOGEN_FONT_PATH to a .ttf to fix this."
    )
    return ImageFont.load_default()


def font_metrics(font: FontLike) -> tuple[float, float]:
    """Return ``(ascent, line_height)`` for the font, tolerating the bitmap fallback."""
    try:
        ascent, descent = font.getmetrics()  # type: ignore[union-attr]
        return float(ascent), float(ascent + descent)
    except AttributeError:
        size = getattr(font, "size", 16)
        return float(size), float(size * 1.2)


def measure(text: str, font: FontLike, draw: ImageDraw.ImageDraw | None = None) -> float:
    """Width of ``text`` in pixels."""
    if draw is not None:
        return float(draw.textlength(text, font=font))
    try:
        return float(font.getlength(text))  # type: ignore[union-attr]
    except AttributeError:
        return float(len(text) * getattr(font, "size", 10) * 0.6)


def wrap_text(
    text: str,
    font: FontLike,
    max_width: float,
    draw: ImageDraw.ImageDraw | None = None,
) -> list[str]:
    """Greedy word wrap to ``max_width`` pixels.

    A word wider than the line box is kept on its own line rather than split, so a long
    URL degrades to overflow instead of nonsense hyphenation.
    """
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if measure(candidate, font, draw) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines
