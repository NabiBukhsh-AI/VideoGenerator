"""Logging setup.

The pipeline is long-running and mostly I/O bound against paid APIs, so logs are the
main way to see what happened and what it cost. Rich is used when available for
readable console output; otherwise this degrades to plain stdlib logging.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

_CONFIGURED = False
_LOGGER_NAME = "videogen"


def _rich_handler() -> logging.Handler | None:
    try:
        from rich.logging import RichHandler
    except ImportError:  # pragma: no cover - rich is a core dep, but stay defensive
        return None
    return RichHandler(rich_tracebacks=True, show_path=False, markup=False)


def configure(level: int | str | None = None, *, force: bool = False) -> None:
    """Configure the `videogen` logger once.

    Level resolution order: explicit argument, then ``VIDEOGEN_LOG_LEVEL``, then INFO.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    if level is None:
        level = os.environ.get("VIDEOGEN_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()

    handler = _rich_handler()
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
        )
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False

    # Vendor SDKs are chatty at DEBUG and leak request bodies; keep them at WARNING.
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the package logger, configuring it on first use."""
    configure()
    if name is None:
        return logging.getLogger(_LOGGER_NAME)
    if name.startswith(_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


class ProgressReporter:
    """Minimal progress sink the pipeline writes to.

    The CLI passes a Rich-backed implementation and Streamlit passes one that drives a
    progress bar, so `pipeline.py` never has to know which front end is running.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._log = logger or get_logger("progress")

    def stage(self, name: str, detail: str = "") -> None:
        self._log.info("%s%s", name, f" - {detail}" if detail else "")

    def update(self, fraction: float, detail: str = "") -> None:
        """Report fractional completion in ``[0, 1]``. No-op for the log-only sink."""

    def warn(self, message: str) -> None:
        self._log.warning(message)

    def metric(self, name: str, value: Any) -> None:
        self._log.debug("metric %s=%s", name, value)
