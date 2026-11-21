"""Logging setup shared by every AtmoGraph NLP component.

The service writes to both the console and a rotating file so failures in
background ingestion (RSS outages, backend timeouts) can be audited later.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from app.config import Settings, get_settings
from app.utils.helpers import ensure_directory

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "transformers", "filelock", "asyncio")
_HANDLER_MARKER = "_atmograph_handler"
_configured = False


def _tag(handler: logging.Handler) -> logging.Handler:
    setattr(handler, _HANDLER_MARKER, True)
    return handler


def _is_ours(handler: logging.Handler) -> bool:
    return bool(getattr(handler, _HANDLER_MARKER, False))


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    """Configure root logging once per process.

    Safe to call multiple times; only handlers created by this module are
    replaced when ``force`` is true.
    """
    global _configured
    if _configured and not force:
        return

    settings = settings or get_settings()
    ensure_directory(settings.log_dir)

    level = getattr(logging, settings.log_level, logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        if _is_ours(handler):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = _tag(logging.StreamHandler(stream=sys.stdout))
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = _tag(
        RotatingFileHandler(
            filename=settings.log_dir / settings.log_file_name,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(level, logging.WARNING))

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (use ``__name__`` at the call site)."""
    return logging.getLogger(name)
