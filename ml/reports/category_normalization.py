"""Source-aware normalization for anonymous and verified categorical values."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

ANONYMIZED_CODE_RE = re.compile(r"^[A-Z]{1,2}$")
MISSING_TOKENS = {
    "",
    "-",
    "--",
    "<na>",
    "n/a",
    "na",
    "nan",
    "nat",
    "none",
    "null",
    "unknown",
    "missing",
    "not available",
    "not_available",
}
CATEGORY_TYPES = ("verified", "anonymized", "missing", "unknown")
ANONYMIZED_SOURCES = {"ic_splc": {"country", "industry", "product_category"}}
VERIFIED_SOURCES = {
    "anonnet": {"country", "product_category"},
    "sampoq": set(),
    "ic_splc": set(),
}
FIELD_LABELS = {
    "country": "Country",
    "industry": "Industry",
    "product_category": "Product category",
}
SOURCE_LABELS = {
    "sampoq": "SAMPOQ",
    "ic_splc": "IC-SPLC",
    "anonnet": "AnonNet",
    "unknown": "Unknown source",
}
NOTES = {
    "country": [
        "IC-SPLC country values are anonymized frequency-ranked codes; their mapping to real countries is not published.",
        "AnonNet location values are preserved as source-provided country names.",
        "SAMPOQ has no country field in the acquired source.",
    ],
    "industry": [
        "IC-SPLC industry values are anonymized frequency-ranked codes; their mapping to real industries is not published.",
        "AnonNet and SAMPOQ have no industry field in the acquired sources.",
        "No verified industry names exist in the current graph.",
    ],
    "product_category": [
        "IC-SPLC product-category values are anonymized frequency-ranked codes; their mapping to real categories is not published.",
        "AnonNet product-category values are preserved as source-provided names.",
        "SAMPOQ has no product-category field in the acquired source.",
    ],
}


def source_label(source: str) -> str:
    return SOURCE_LABELS.get(source, source.replace("_", " ").title() if source else "Unknown source")


def clean_raw(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return None
    return text


def classify_category(value: Any, field: str, source: Any) -> dict[str, Any]:
    field = field.strip().lower()
    if field not in FIELD_LABELS:
        raise ValueError(f"Unsupported category field: {field}")
    raw = clean_raw(value)
    source_name = str(source or "unknown").strip().lower() or "unknown"
    label = FIELD_LABELS[field]
    if raw is None:
        return {
            "raw_value": None,
            "normalized_value": "Unknown / Missing",
            "category_type": "missing",
            "source": source_name,
        }
    if source_name in ANONYMIZED_SOURCES and field in ANONYMIZED_SOURCES[source_name]:
        if ANONYMIZED_CODE_RE.fullmatch(raw):
            normalized = f"{label} Code {raw}"
        else:
            normalized = f"Anonymized {label.lower()} {raw}"
        return {
            "raw_value": raw,
            "normalized_value": normalized,
            "category_type": "anonymized",
            "source": source_name,
        }
    if field in VERIFIED_SOURCES.get(source_name, set()) and not ANONYMIZED_CODE_RE.fullmatch(raw):
        return {
            "raw_value": raw,
            "normalized_value": raw,
            "category_type": "verified",
            "source": source_name,
        }
    return {
        "raw_value": raw,
        "normalized_value": f"Unrecognized {label.lower()} {raw}",
        "category_type": "unknown",
        "source": source_name,
    }


def aggregate_categories(
    values: Sequence[Any],
    sources: Sequence[Any],
    field: str,
    top_n: int = 15,
    include_all_values: bool = False,
) -> dict[str, Any]:
    if len(values) != len(sources):
        raise ValueError("values and sources must have the same length")
    field = field.strip().lower()
    if field not in FIELD_LABELS:
        raise ValueError(f"Unsupported category field: {field}")

    total = len(values)
    counts: Counter[str] = Counter()
    distinct: dict[str, set[str]] = {category_type: set() for category_type in CATEGORY_TYPES}
    value_counts: Counter[tuple[str | None, str, str, str]] = Counter()
    source_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    raw_counts: Counter[str] = Counter()
    raw_source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    missing_by_source: Counter[str] = Counter()

    for value, source in zip(values, sources):
        classified = classify_category(value, field, source)
        category_type = classified["category_type"]
        source_name = classified["source"]
        raw = classified["raw_value"]
        counts[category_type] += 1
        source_type_counts[source_name][category_type] += 1
        if raw is not None:
            distinct[category_type].add(raw)
            value_counts[(raw, source_name, category_type, classified["normalized_value"])] += 1
            raw_counts[raw] += 1
            raw_source_counts[raw][source_name] += 1
        if category_type == "missing":
            missing_by_source[source_name] += 1

    def entries_for(category_type: str) -> list[dict[str, Any]]:
        family_count = counts.get(category_type, 0)
        entries = [
            {
                "raw_value": raw,
                "normalized_value": normalized,
                "category_type": entry_type,
                "source": source_name,
                "count": count,
                "share_of_total": round(count / total, 8) if total else 0.0,
                "share_of_type": round(count / family_count, 8) if family_count else 0.0,
            }
            for (raw, source_name, entry_type, normalized), count in value_counts.items()
            if entry_type == category_type
        ]
        entries.sort(key=lambda entry: (-entry["count"], str(entry["normalized_value"]).lower()))
        return entries

    top_verified = entries_for("verified")[:top_n]
    top_anonymized = entries_for("anonymized")[:top_n]
    top_unknown = entries_for("unknown")[:top_n]
    source_names = sorted(source_type_counts)
    by_source = {
        source_name: {
            "total": sum(source_type_counts[source_name].values()),
            **{category_type: source_type_counts[source_name].get(category_type, 0) for category_type in CATEGORY_TYPES},
        }
        for source_name in source_names
    }
    result: dict[str, Any] = {
        "field": field,
        "label": FIELD_LABELS[field],
        "total_nodes": total,
        "counts": {
            "total": total,
            **{category_type: counts.get(category_type, 0) for category_type in CATEGORY_TYPES},
        },
        "distinct": {
            category_type: len(distinct[category_type]) for category_type in CATEGORY_TYPES
        },
        "top_verified": top_verified,
        "top_anonymized": top_anonymized,
        "top_unknown": top_unknown,
        "missing_by_source": {
            source_name: missing_by_source[source_name] for source_name in sorted(missing_by_source)
        },
        "by_source": by_source,
        "notes": NOTES.get(field, []),
    }
    if include_all_values:
        result["raw_distinct_values"] = [
            {
                "raw_value": raw,
                "count": count,
                "sources": {
                    source_name: raw_source_counts[raw][source_name]
                    for source_name in sorted(raw_source_counts[raw])
                },
            }
            for raw, count in sorted(raw_counts.items(), key=lambda item: (-item[1], item[0].lower()))
        ]
        result["normalized_distinct_values"] = entries_for("verified") + entries_for("anonymized") + entries_for("unknown")
    return result
