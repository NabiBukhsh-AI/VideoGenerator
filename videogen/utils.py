"""Small shared helpers: retries, hashing, slugs, timing, parallel mapping."""

from __future__ import annotations

import functools
import hashlib
import json
import random
import re
import time
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypeVar

from .errors import ProviderError
from .logging import get_logger

T = TypeVar("T")
R = TypeVar("R")

log = get_logger("utils")

# Characters that trip up text-to-speech engines or shell/ffmpeg quoting.
_SMART_PUNCTUATION = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2026": "...",
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",
    "`": "'",
}


def normalize_text(text: str) -> str:
    """Replace smart punctuation and collapse whitespace.

    The original code called ``str.replace`` and threw the result away, so every
    downstream consumer still saw curly quotes. Returning the value is the fix.
    """
    for bad, good in _SMART_PUNCTUATION.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def slugify(value: str, *, max_length: int = 60) -> str:
    """Turn arbitrary text into a filesystem- and URL-safe slug."""
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    value = re.sub(r"[-\s]+", "-", value)
    return value[:max_length].strip("-") or "untitled"


def stable_hash(*parts: Any, length: int = 12) -> str:
    """Deterministic short hash over JSON-serialisable parts.

    Used as a cache key so a re-run with identical inputs can skip a paid API call.
    """
    hasher = hashlib.sha256()
    for part in parts:
        if isinstance(part, (str, bytes)):
            blob = part.encode("utf-8") if isinstance(part, str) else part
        elif isinstance(part, Path):
            blob = str(part).encode("utf-8")
        else:
            blob = json.dumps(part, sort_keys=True, default=str).encode("utf-8")
        hasher.update(blob)
        hasher.update(b"\x00")
    return hasher.hexdigest()[:length]


def file_hash(path: Path, *, length: int = 12) -> str:
    """Content hash of a file, streamed so large PDFs/videos stay cheap."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:length]


def retry(
    attempts: int = 4,
    *,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    jitter: bool = True,
):
    """Retry a callable with exponential backoff and full jitter.

    Only retries ``ProviderError`` instances that declare ``retryable=True``; every
    other exception propagates on the first failure so genuine bugs surface fast.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if isinstance(exc, ProviderError) and not exc.retryable:
                        raise
                    last = exc
                    if attempt == attempts:
                        break
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    if jitter:
                        delay = random.uniform(0, delay)
                    log.warning(
                        "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                        func.__name__,
                        attempt,
                        attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            assert last is not None
            raise last

        return wrapper

    return decorator


def parallel_map(
    func: Callable[[T], R],
    items: Sequence[T],
    *,
    max_workers: int = 4,
    on_result: Callable[[int, R], None] | None = None,
) -> list[R]:
    """Run ``func`` over ``items`` in a thread pool, preserving input order.

    Image generation and TTS are network-bound and independent per scene, so doing
    them concurrently is most of the wall-clock win over the original sequential loop.
    The first exception is re-raised once all in-flight work settles.
    """
    if not items:
        return []
    workers = max(1, min(max_workers, len(items)))
    if workers == 1:
        results = []
        for index, item in enumerate(items):
            value = func(item)
            if on_result:
                on_result(index, value)
            results.append(value)
        return results

    results_by_index: dict[int, R] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="videogen") as pool:
        futures = {pool.submit(func, item): index for index, item in enumerate(items)}
        error: BaseException | None = None
        for future in as_completed(futures):
            index = futures[future]
            try:
                value = future.result()
            except BaseException as exc:
                error = error or exc
                continue
            results_by_index[index] = value
            if on_result:
                on_result(index, value)
        if error is not None:
            raise error
    return [results_by_index[i] for i in range(len(items))]


def chunked(items: Iterable[T], size: int) -> Iterable[list[T]]:
    """Yield fixed-size chunks from an iterable."""
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def format_duration(milliseconds: float) -> str:
    """Render milliseconds as ``M:SS.mmm`` for logs and progress output."""
    total_seconds, ms = divmod(int(milliseconds), 1000)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}.{ms:03d}"


class Timer:
    """Context manager that logs how long a stage took."""

    def __init__(self, label: str, logger: Any = None) -> None:
        self.label = label
        self.log = logger or log
        self.elapsed_ms = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        self.log.debug("%s took %s", self.label, format_duration(self.elapsed_ms))
