"""Lazy, process-wide spaCy model loading.

The model is loaded once and reused.  No download is attempted at runtime: if
the configured package is missing the service raises a clear error instead of
silently reaching the network.  Inference is serialized because a single spaCy
pipeline is not documented as thread-safe for concurrent ``nlp(text)`` calls.
"""

from __future__ import annotations

import importlib.util
import threading
from typing import Any

import spacy
from spacy.language import Language

from app.utils.logger import get_logger

logger = get_logger(__name__)


class NlpModelError(RuntimeError):
    """Raised when the configured spaCy model cannot be loaded."""


_MODEL_CACHE: dict[str, Language] = {}
_MODEL_LOCK = threading.RLock()
_INFERENCE_LOCK = threading.RLock()


def is_model_available(model_name: str) -> bool:
    """Return whether the model package is installed (without loading it)."""
    if not model_name:
        return False
    try:
        if spacy.util.is_package(model_name):
            return True
        return importlib.util.find_spec(model_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def load_spacy_model(model_name: str) -> Language:
    """Load a spaCy pipeline or raise :class:`NlpModelError`."""
    try:
        logger.info("Loading spaCy model '%s'", model_name)
        return spacy.load(model_name)
    except OSError as exc:
        raise NlpModelError(
            f"spaCy model '{model_name}' is not installed. Install it with "
            f"`.venv\\Scripts\\python.exe -m spacy download {model_name}` "
            "and restart the service."
        ) from exc


def get_nlp(model_name: str = "en_core_web_sm") -> Language:
    """Return the cached pipeline, loading it on first use."""
    with _MODEL_LOCK:
        pipeline = _MODEL_CACHE.get(model_name)
        if pipeline is None:
            pipeline = load_spacy_model(model_name)
            _MODEL_CACHE[model_name] = pipeline
        return pipeline


def is_model_loaded(model_name: str) -> bool:
    with _MODEL_LOCK:
        return model_name in _MODEL_CACHE


def get_loaded_model(model_name: str) -> Language | None:
    with _MODEL_LOCK:
        return _MODEL_CACHE.get(model_name)


def model_metadata(model_name: str) -> dict[str, Any] | None:
    """Return the loaded model's metadata, or ``None`` if it is not loaded."""
    pipeline = get_loaded_model(model_name)
    if pipeline is None:
        return None
    return dict(pipeline.meta)


def parse_text(pipeline: Language, text: str):
    """Run the pipeline under a global lock so concurrent requests are safe."""
    with _INFERENCE_LOCK:
        return pipeline(text)


def reset_model_cache() -> None:
    """Drop cached pipelines (used by tests that need a clean loader)."""
    with _MODEL_LOCK:
        _MODEL_CACHE.clear()


__all__ = [
    "NlpModelError",
    "get_loaded_model",
    "get_nlp",
    "is_model_available",
    "is_model_loaded",
    "load_spacy_model",
    "model_metadata",
    "parse_text",
    "reset_model_cache",
]
