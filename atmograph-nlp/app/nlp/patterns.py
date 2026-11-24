"""Configurable spaCy Matcher patterns for supply-chain entity types.

Phrase patterns cover the gazetteers in :mod:`app.nlp.taxonomy`; token patterns
add generic structures such as ``Port of <name>`` and ``<ProperNoun> Terminal``.
Extra patterns can be supplied as JSON through ``ATMOGRAPH_NLP_PATTERNS_PATH``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from app.nlp.taxonomy import (
    ENTITY_TYPES,
    FACILITY_NAMES,
    INDUSTRY_NAMES,
    MATERIAL_NAMES,
    ORGANIZATION_NAMES,
    PORT_NAMES,
    PRODUCT_NAMES,
    RULE_LABELS,
    RULE_LABEL_PREFIX,
)

Pattern = dict[str, Any]


class PatternConfigError(ValueError):
    """Raised when a custom pattern file is missing or malformed."""


_FACILITY_SUFFIXES = (
    "terminal",
    "terminals",
    "plant",
    "plants",
    "factory",
    "factories",
    "mill",
    "mills",
    "warehouse",
    "warehouses",
    "depot",
    "depots",
    "refinery",
    "refineries",
    "smelter",
    "smelters",
    "airport",
    "airports",
    "port",
    "yard",
    "yards",
    "hub",
    "hubs",
    "park",
    "parks",
    "centre",
    "centres",
    "center",
    "centers",
    "facility",
    "facilities",
)


def _phrase(label: str, text: str) -> Pattern:
    return {"label": label, "pattern": text}


def _default_patterns() -> list[Pattern]:
    patterns: list[Pattern] = []

    for name in PORT_NAMES:
        patterns.append(_phrase(f"{RULE_LABEL_PREFIX}PORT", name))

    for name in FACILITY_NAMES:
        patterns.append(_phrase(f"{RULE_LABEL_PREFIX}FACILITY", name))

    for name in MATERIAL_NAMES:
        patterns.append(_phrase(f"{RULE_LABEL_PREFIX}MATERIAL", name))

    for name in INDUSTRY_NAMES:
        patterns.append(_phrase(f"{RULE_LABEL_PREFIX}INDUSTRY", name))

    for name in PRODUCT_NAMES:
        patterns.append(_phrase(f"{RULE_LABEL_PREFIX}PRODUCT", name))

    for name in ORGANIZATION_NAMES:
        patterns.append(_phrase(f"{RULE_LABEL_PREFIX}ORGANIZATION", name))

    # Generic port structures: "Port of X", "Port of the X", "X Port".
    patterns.extend(
        [
            {
                "label": f"{RULE_LABEL_PREFIX}PORT",
                "pattern": [
                    {"LOWER": "port"},
                    {"LOWER": "of"},
                    {"LOWER": "the"},
                    {"POS": "PROPN", "OP": "+"},
                ],
            },
            {
                "label": f"{RULE_LABEL_PREFIX}PORT",
                "pattern": [
                    {"LOWER": "port"},
                    {"LOWER": "of"},
                    {"POS": "PROPN", "OP": "+"},
                ],
            },
            {
                "label": f"{RULE_LABEL_PREFIX}PORT",
                "pattern": [
                    {"POS": "PROPN", "OP": "+"},
                    {"LOWER": "port"},
                ],
            },
        ]
    )

    # Generic facility structures: "Kaohsiung Terminal", "Penang Plant", ...
    patterns.append(
        {
            "label": f"{RULE_LABEL_PREFIX}FACILITY",
            "pattern": [
                {"POS": "PROPN", "OP": "+"},
                {"LOWER": {"IN": list(_FACILITY_SUFFIXES)}},
            ],
        }
    )

    return patterns


DEFAULT_PATTERNS: tuple[Pattern, ...] = tuple(_default_patterns())


def _normalise_label(label: Any) -> str:
    if not isinstance(label, str) or not label.strip():
        raise PatternConfigError("Every custom pattern needs a non-empty 'label'.")

    candidate = label.strip().upper()
    if candidate.startswith(RULE_LABEL_PREFIX):
        if candidate not in RULE_LABELS:
            raise PatternConfigError(
                f"Unknown entity label '{label}'. Valid labels: "
                f"{', '.join(ENTITY_TYPES)}."
            )
        return candidate

    if candidate not in ENTITY_TYPES:
        raise PatternConfigError(
            f"Unknown entity label '{label}'. Valid labels: {', '.join(ENTITY_TYPES)}."
        )
    return f"{RULE_LABEL_PREFIX}{candidate}"


def _normalise_pattern(pattern: Any) -> str | list[dict[str, Any]]:
    if isinstance(pattern, str):
        text = pattern.strip()
        if not text:
            raise PatternConfigError("Custom phrase patterns cannot be empty.")
        return text
    if isinstance(pattern, dict):
        pattern = [pattern]
    if isinstance(pattern, list) and pattern and all(isinstance(item, dict) for item in pattern):
        return pattern
    raise PatternConfigError(
        "A custom 'pattern' must be a non-empty string or a list of spaCy token-pattern objects."
    )


def load_custom_patterns(path: Path | str | None) -> list[Pattern]:
    """Load and validate an optional JSON pattern file.

    Accepted shapes::

        [{"label": "PORT", "pattern": "Port of My Town"}]
        {"patterns": [{"label": "MATERIAL", "pattern": [{"LOWER": "cork"}]}]}
    """
    if path is None:
        return []

    pattern_path = Path(path)
    if not pattern_path.exists():
        raise PatternConfigError(f"Pattern file '{pattern_path}' does not exist.")
    try:
        payload = json.loads(pattern_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PatternConfigError(f"Cannot read pattern file '{pattern_path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PatternConfigError(
            f"Pattern file '{pattern_path}' contains invalid JSON "
            f"(line {exc.lineno}, column {exc.colno})."
        ) from exc

    if isinstance(payload, dict):
        payload = payload.get("patterns")
    if not isinstance(payload, list):
        raise PatternConfigError(
            f"Pattern file '{pattern_path}' must contain a JSON array or a 'patterns' array."
        )

    normalised: list[Pattern] = []
    for position, item in enumerate(payload):
        if not isinstance(item, dict) or "label" not in item or "pattern" not in item:
            raise PatternConfigError(
                f"Pattern entry {position} in '{pattern_path}' must be an object with "
                "'label' and 'pattern'."
            )
        normalised.append(
            {
                "label": _normalise_label(item["label"]),
                "pattern": _normalise_pattern(item["pattern"]),
            }
        )
    return normalised


def build_patterns(custom_path: Path | str | None = None) -> list[Pattern]:
    """Return built-in patterns followed by validated custom patterns."""
    combined: Iterable[Pattern] = (*DEFAULT_PATTERNS, *load_custom_patterns(custom_path))
    unique: list[Pattern] = []
    seen: set[tuple[str, str]] = set()
    for pattern in combined:
        signature = (
            str(pattern["label"]),
            json.dumps(pattern["pattern"], sort_keys=True, ensure_ascii=False),
        )
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(pattern)
    return unique


__all__ = [
    "DEFAULT_PATTERNS",
    "Pattern",
    "PatternConfigError",
    "build_patterns",
    "load_custom_patterns",
]
