"""Web article ingestion.

New capability: point the tool at a URL and it builds the video from that page.
Extraction prefers the semantic container (``<article>``, ``<main>``, or the densest
block of paragraphs) so navigation, cookie banners and footers stay out of the script.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from ..errors import DependencyError, IngestError
from ..logging import get_logger
from ..models import SourceDocument
from ..utils import normalize_text
from .base import SourceInput, SourceLoader

log = get_logger("ingest.web")

_USER_AGENT = "Mozilla/5.0 (compatible; videogen/2.0; +https://github.com/NabiBukhsh-AI/VideoGenerator)"
_STRIP_TAGS = ("script", "style", "nav", "header", "footer", "aside", "form", "noscript",
               "iframe", "svg", "figure")


class WebLoader(SourceLoader):
    kind = "web"

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def can_handle(self, source: SourceInput) -> bool:
        if not isinstance(source, str):
            return False
        parsed = urlparse(source.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def load(self, source: SourceInput) -> SourceDocument:
        url = str(source).strip()
        html = self._fetch(url)
        title, text = self._extract(html, url)

        if len(text.split()) < 40:
            raise IngestError(
                f"Only extracted {len(text.split())} words from {url}.",
                hint=(
                    "The page may be JavaScript-rendered or paywalled. "
                    "Copy the article text and pass it directly instead."
                ),
            )

        log.info("Extracted %d words from %s", len(text.split()), url)
        return SourceDocument(
            text=text,
            name=title or urlparse(url).netloc,
            kind=self.kind,
            metadata={"url": url, "title": title},
        )

    def _fetch(self, url: str) -> str:
        import httpx

        try:
            response = httpx.get(
                url,
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": _USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            )
            response.raise_for_status()
        except Exception as exc:
            raise IngestError(f"Could not fetch {url}: {exc}") from exc

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            raise IngestError(
                f"{url} returned {content_type or 'an unknown content type'}, not an HTML page."
            )
        return response.text

    def _extract(self, html: str, url: str) -> tuple[str, str]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise DependencyError(
                "beautifulsoup4 is required to read web pages.",
                hint="pip install 'videogen[web]'",
            ) from exc

        soup = BeautifulSoup(html, _parser())

        for tag in soup(list(_STRIP_TAGS)):
            tag.decompose()

        title = ""
        if soup.title and soup.title.string:
            title = normalize_text(soup.title.string)
        if heading := soup.find("h1"):
            title = normalize_text(heading.get_text()) or title

        container = soup.find("article") or soup.find("main") or _densest_block(soup)
        if container is None:
            container = soup.body or soup

        paragraphs = [
            normalize_text(p.get_text(" "))
            for p in container.find_all(["p", "li", "h2", "h3", "blockquote"])
        ]
        # Drop one-liners: they are captions, bylines and share prompts, not prose.
        body = " ".join(p for p in paragraphs if len(p.split()) >= 6)

        if not body:
            body = normalize_text(container.get_text(" "))

        log.debug("Extracted article %r from %s", title[:60], url)
        return title, body


def _parser() -> str:
    """Prefer lxml when installed; html.parser always exists."""
    try:
        import lxml  # noqa: F401

        return "lxml"
    except ImportError:
        return "html.parser"


def _densest_block(soup):
    """Fall back to the div/section containing the most paragraph text."""
    best, best_score = None, 0
    for node in soup.find_all(["div", "section"]):
        score = sum(len(p.get_text()) for p in node.find_all("p", recursive=False))
        if score > best_score:
            best, best_score = node, score
    return best


class HTMLFileLoader(SourceLoader):
    """A saved .html file on disk."""

    kind = "htmlfile"
    extensions = (".html", ".htm")

    def can_handle(self, source: SourceInput) -> bool:
        if isinstance(source, Path):
            return source.suffix.lower() in self.extensions
        if isinstance(source, str):
            path = Path(source)
            return path.suffix.lower() in self.extensions and path.exists()
        return False

    def load(self, source: SourceInput) -> SourceDocument:
        path = Path(str(source))
        html = path.read_text(encoding="utf-8", errors="replace")
        title, text = WebLoader()._extract(html, str(path))
        if not text.strip():
            raise IngestError(f"No readable text in {path.name}.")
        return SourceDocument(
            text=text, name=title or path.stem, kind=self.kind, metadata={"path": str(path)}
        )
