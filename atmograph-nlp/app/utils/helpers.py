"""Small, dependency-free helpers used across the NLP module."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> datetime:
    """Timezone-aware UTC timestamp (all API timestamps are UTC)."""
    return datetime.now(timezone.utc)


def ensure_directory(path: Path) -> Path:
    """Create ``path`` (including parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path
