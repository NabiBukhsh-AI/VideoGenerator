"""Source loader interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO

from ..models import SourceDocument

#: Anything a loader accepts: a path, a URL string, raw text, or an open file handle
#: (which is what Streamlit's uploader hands you).
SourceInput = str | Path | IO[bytes]


class SourceLoader(ABC):
    """Turns some input into normalised text."""

    #: Identifier recorded on the resulting document.
    kind: str = "text"

    #: File extensions this loader claims, lowercase and dotted.
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def can_handle(self, source: SourceInput) -> bool:
        """True when this loader recognises ``source``."""

    @abstractmethod
    def load(self, source: SourceInput) -> SourceDocument:
        """Extract text, raising ``IngestError`` if nothing usable is found."""

    @staticmethod
    def _name_of(source: SourceInput) -> str:
        if isinstance(source, Path):
            return source.name
        if isinstance(source, str):
            return source[:60]
        return getattr(source, "name", "upload")
