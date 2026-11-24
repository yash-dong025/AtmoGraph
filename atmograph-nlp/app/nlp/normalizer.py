"""Entity-name cleaning, alias resolution and canonical keys.

The extractor keeps the original surface text for offsets and provenance, but
also stores a human-readable ``normalized_name`` and a punctuation/case-folded
key for duplicate checks.  Country abbreviations, material spelling variants and
organisation acronyms are resolved through small, explicit alias tables.
"""

from __future__ import annotations

import re
import unicodedata

from app.nlp.taxonomy import (
    COUNTRY_ALIASES,
    COUNTRY_NAMES,
    FACILITY_NAMES,
    INDUSTRY_NAMES,
    KNOWN_LOCATION_NAMES,
    MATERIAL_ALIASES,
    MATERIAL_NAMES,
    ORGANIZATION_ALIASES,
    ORGANIZATION_NAMES,
    PORT_NAMES,
    PRODUCT_NAMES,
    EntityType,
)

_WHITESPACE_RE = re.compile(r"\s+")
_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_POSSESSIVE_RE = re.compile(r"(?:'s|s')\s*$", re.IGNORECASE)
_EDGE_PUNCTUATION = " \t\n\r.,;:!?()[]{}<>\"'“”‘’«»"


def _clean_display(value: str | None) -> str:
    """Normalize Unicode, whitespace, articles and edge punctuation."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = _LEADING_ARTICLE_RE.sub("", text)
    text = _POSSESSIVE_RE.sub("", text)
    text = text.strip(_EDGE_PUNCTUATION)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _base_key(value: str | None) -> str:
    """Case-folded key before alias resolution (no display formatting)."""
    if value is None:
        return ""
    raw = str(value)
    text = unicodedata.normalize("NFKC", raw).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = _LEADING_ARTICLE_RE.sub("", text)
    if re.search(r"(?:'s|’s)\s*$", raw, flags=re.IGNORECASE):
        text = re.sub(r"\s+s$", "", text)
    return text


_ALIAS_DISPLAY: dict[str, str] = {}
for _alias, _canonical in (
    *COUNTRY_ALIASES.items(),
    *MATERIAL_ALIASES.items(),
    *ORGANIZATION_ALIASES.items(),
):
    _ALIAS_DISPLAY.setdefault(_base_key(_alias), _canonical)

_CANONICAL_DISPLAY: dict[str, str] = {}
for _name in (
    *COUNTRY_NAMES,
    *KNOWN_LOCATION_NAMES,
    *PORT_NAMES,
    *FACILITY_NAMES,
    *MATERIAL_NAMES,
    *INDUSTRY_NAMES,
    *PRODUCT_NAMES,
    *ORGANIZATION_NAMES,
):
    _CANONICAL_DISPLAY.setdefault(_base_key(_name), _name)

_KEY_ALIASES: dict[str, str] = {
    _base_key(alias): _base_key(canonical) for alias, canonical in _ALIAS_DISPLAY.items()
}


def canonical_key(value: str | None) -> str:
    """Return a stable, case/punctuation-insensitive key for an entity name."""
    key = _base_key(value)
    return _KEY_ALIASES.get(key, key)


def normalize_entity_name(value: str | None, entity_type: EntityType | None = None) -> str:
    """Return a clean display name with known aliases resolved.

    ``entity_type`` is accepted for callers that want to make the intent
    explicit; alias resolution is deliberately shared across types so ``U.S.``
    and ``United States`` always compare equal.
    """
    cleaned = _clean_display(value)
    if not cleaned:
        return ""
    key = _base_key(cleaned)
    return _ALIAS_DISPLAY.get(key) or _CANONICAL_DISPLAY.get(key) or cleaned


__all__ = ["canonical_key", "normalize_entity_name"]
