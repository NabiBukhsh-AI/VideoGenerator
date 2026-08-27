"""Loader dispatch.

Order matters: the most specific matcher wins, and ``RawTextLoader`` is last because
it accepts any string.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import IngestError
from ..logging import get_logger
from ..models import SourceDocument
from .base import SourceInput, SourceLoader
from .pdf import PDFLoader
from .plaintext import RawTextLoader, TextFileLoader, UploadLoader
from .web import HTMLFileLoader, WebLoader

log = get_logger("ingest")


def _loaders() -> list[SourceLoader]:
    return [
        WebLoader(),
        PDFLoader(),
        TextFileLoader(),
        HTMLFileLoader(),
        UploadLoader(),
        RawTextLoader(),
    ]


def loader_for(source: SourceInput) -> SourceLoader:
    """Return the first loader that recognises ``source``."""
    for loader in _loaders():
        try:
            if loader.can_handle(source):
                return loader
        except Exception as exc:
            log.debug("%s.can_handle raised %s", type(loader).__name__, exc)

    raise IngestError(
        f"No loader can handle {_describe(source)}.",
        hint="Supported: PDF, .txt/.md, .html, an http(s) URL, or raw text.",
    )


def load_source(source: SourceInput, *, max_words: int | None = None) -> SourceDocument:
    """Load ``source`` into a normalised document, optionally truncated."""
    loader = loader_for(source)
    log.debug("Using %s for %s", type(loader).__name__, _describe(source))
    document = loader.load(source)

    if max_words is not None:
        original = document.word_count
        document = document.truncate(max_words)
        if document.word_count < original:
            log.info(
                "Truncated source from %d to %d words to fit the model context window.",
                original,
                document.word_count,
            )
    return document


def _describe(source: SourceInput) -> str:
    if isinstance(source, Path):
        return f"path {source.name!r}"
    if isinstance(source, str):
        preview = source if len(source) <= 60 else source[:57] + "..."
        return f"input {preview!r}"
    return f"upload {getattr(source, 'name', 'stream')!r}"
