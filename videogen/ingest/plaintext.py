"""Plain text, Markdown and raw-string ingestion."""

from __future__ import annotations

import re
from pathlib import Path

from ..errors import IngestError
from ..models import SourceDocument
from ..utils import normalize_text
from .base import SourceInput, SourceLoader

_TEXT_EXTENSIONS = (".txt", ".md", ".markdown", ".rst", ".text")


class TextFileLoader(SourceLoader):
    kind = "textfile"
    extensions = _TEXT_EXTENSIONS

    def can_handle(self, source: SourceInput) -> bool:
        if isinstance(source, Path):
            return source.suffix.lower() in _TEXT_EXTENSIONS
        if isinstance(source, str):
            path = Path(source)
            return path.suffix.lower() in _TEXT_EXTENSIONS and path.exists()
        return False

    def load(self, source: SourceInput) -> SourceDocument:
        path = Path(str(source))
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise IngestError(f"Could not read {path}: {exc}") from exc

        text = normalize_text(_strip_markdown(raw) if path.suffix.lower() in
                              (".md", ".markdown") else raw)
        if not text:
            raise IngestError(f"{path.name} is empty.")

        return SourceDocument(text=text, name=path.stem, kind=self.kind)


class RawTextLoader(SourceLoader):
    """Last-resort loader: treats the input as the source material itself."""

    kind = "raw"

    def can_handle(self, source: SourceInput) -> bool:
        return isinstance(source, str)

    def load(self, source: SourceInput) -> SourceDocument:
        text = normalize_text(str(source))
        if len(text.split()) < 5:
            raise IngestError(
                "Source text is too short to build a video from.",
                hint="Provide at least a couple of sentences.",
            )
        return SourceDocument(text=text, name="pasted-text", kind=self.kind)


class UploadLoader(SourceLoader):
    """Handles an in-memory upload (Streamlit) whose name says it is text."""

    kind = "upload"
    extensions = _TEXT_EXTENSIONS

    def can_handle(self, source: SourceInput) -> bool:
        if isinstance(source, (str, Path)):
            return False
        name = getattr(source, "name", "")
        return Path(name).suffix.lower() in _TEXT_EXTENSIONS

    def load(self, source: SourceInput) -> SourceDocument:
        source.seek(0)  # type: ignore[union-attr]
        blob = source.read()  # type: ignore[union-attr]
        raw = blob.decode("utf-8", errors="replace") if isinstance(blob, bytes) else str(blob)
        name = Path(getattr(source, "name", "upload")).stem
        text = normalize_text(_strip_markdown(raw))
        if not text:
            raise IngestError(f"Uploaded file {name!r} is empty.")
        return SourceDocument(text=text, name=name, kind=self.kind)


def _strip_markdown(text: str) -> str:
    """Remove markup that would otherwise be read aloud as literal characters."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)  # fenced code
    text = re.sub(r"`([^`]*)`", r"\1", text)  # inline code
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)  # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links -> label
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)  # blockquotes
    text = re.sub(r"(\*\*|__|\*|_)", "", text)  # emphasis
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)  # bullets
    text = re.sub(r"^\s*\|.*\|\s*$", " ", text, flags=re.MULTILINE)  # tables
    return text
