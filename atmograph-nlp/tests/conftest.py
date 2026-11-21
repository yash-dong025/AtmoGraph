"""Shared pytest setup.

Environment variables are set before the FastAPI app (and therefore the cached
settings object) is imported, so tests never write into the project log folder.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TEST_LOG_DIR = Path(tempfile.gettempdir()) / "atmograph-nlp-test-logs"
os.environ.setdefault("ATMOGRAPH_ENVIRONMENT", "testing")
os.environ.setdefault("ATMOGRAPH_LOG_DIR", str(_TEST_LOG_DIR))
os.environ.setdefault("ATMOGRAPH_LOG_LEVEL", "WARNING")
os.environ.setdefault("ATMOGRAPH_GRAPH_SYNC_ENABLED", "false")
