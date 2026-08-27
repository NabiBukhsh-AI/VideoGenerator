"""PDF ingestion via PyMuPDF.

Improvements over the original ``extract_text_from_pdf``:

* handles file handles (Streamlit uploads) as well as paths;
* strips running headers/footers that repeat on most pages, which otherwise dominate
  the extracted text and mislead the script generator;
* repairs hyphenated line breaks so "extraordi-\\nnary" is one word;
* raises a clear error for scanned PDFs with no text layer, instead of silently
  handing an empty string to the model.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import IO

from ..errors import DependencyError, IngestError
from ..logging import get_logger
from ..models import SourceDocument
from ..utils import normalize_text
from .base import SourceInput, SourceLoader

log = get_logger("ingest.pdf")

_PDF_MAGIC = b"%PDF"
# A line appearing on this fraction of pages is chrome, not content.
_BOILERPLATE_RATIO = 0.6
_MIN_PAGES_FOR_BOILERPLATE = 4

# Matches a line that is only a page number, optionally bracketed by dashes.
# PDFs use either a hyphen or an EN DASH, so the class carries both; the en dash is
# built with chr() to keep this file pure ASCII.
_DASHES = chr(0x2013) + "-"
_PAGE_NUMBER = re.compile(f"^[\\s{_DASHES}]*\\d{{1,4}}[\\s{_DASHES}]*$")


class PDFLoader(SourceLoader):
    kind = "pdf"
    extensions = (".pdf",)

    def can_handle(self, source: SourceInput) -> bool:
        if isinstance(source, Path):
            return source.suffix.lower() == ".pdf"
        if isinstance(source, str):
            return source.lower().endswith(".pdf") and Path(source).exists()
        return _sniff_pdf(source)

    def load(self, source: SourceInput) -> SourceDocument:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise DependencyError(
                "PyMuPDF is required to read PDFs.",
                hint="pip install 'videogen[pdf]'",
            ) from exc

        name = self._name_of(source)
        try:
            if isinstance(source, (str, Path)):
                document = fitz.open(str(source))
            else:
                source.seek(0)
                document = fitz.open(stream=source.read(), filetype="pdf")
        except Exception as exc:
            raise IngestError(
                f"Could not open PDF {name!r}: {exc}",
                hint="Confirm the file is a valid, non-encrypted PDF.",
            ) from exc

        with document:
            if document.is_encrypted and not document.authenticate(""):
                raise IngestError(
                    f"PDF {name!r} is password protected.",
                    hint="Decrypt it first, or paste the text directly.",
                )
            pages = [page.get_text("text") or "" for page in document]
            page_count = document.page_count

        cleaned = _strip_boilerplate(pages)
        text = normalize_text(_dehyphenate("\n".join(cleaned)))

        if len(text.split()) < 20:
            raise IngestError(
                f"PDF {name!r} yielded almost no text ({len(text.split())} words).",
                hint=(
                    "This is usually a scanned PDF with no text layer. Run it through OCR "
                    "(e.g. ocrmypdf) or paste the text directly."
                ),
            )

        log.info("Extracted %d words from %d pages of %s", len(text.split()), page_count, name)
        return SourceDocument(
            text=text,
            name=Path(name).stem or "document",
            kind=self.kind,
            metadata={"pages": page_count},
        )


def _sniff_pdf(handle: IO[bytes]) -> bool:
    """Detect a PDF by magic bytes, restoring the stream position afterwards."""
    try:
        position = handle.tell()
        head = handle.read(5)
        handle.seek(position)
        return head.startswith(_PDF_MAGIC)
    except Exception:
        return False


def _strip_boilerplate(pages: list[str]) -> list[str]:
    """Drop short lines that repeat across most pages (headers, footers, page numbers)."""
    if len(pages) < _MIN_PAGES_FOR_BOILERPLATE:
        return pages

    counts: Counter[str] = Counter()
    for page in pages:
        # Only the first and last few lines can be running chrome.
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        for line in lines[:3] + lines[-3:]:
            if len(line) <= 80:
                counts[line] += 1

    threshold = max(2, int(len(pages) * _BOILERPLATE_RATIO))
    boilerplate = {line for line, count in counts.items() if count >= threshold}
    # Bare page numbers repeat with different values, so match them by shape too.
    numeric = _PAGE_NUMBER

    if boilerplate:
        log.debug("Dropping %d boilerplate line(s): %s", len(boilerplate), sorted(boilerplate)[:3])

    return [
        "\n".join(
            line
            for line in page.splitlines()
            if line.strip() not in boilerplate and not numeric.match(line.strip())
        )
        for page in pages
    ]


def _dehyphenate(text: str) -> str:
    """Join words split across a line break by a hyphen."""
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
