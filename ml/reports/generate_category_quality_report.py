"""Generate reproducible country/industry category-quality reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ml.reports.category_normalization import aggregate_categories, source_label

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NODES_PATH = PROJECT_ROOT / "datasets" / "processed" / "graph" / "nodes.csv"
DATASET_METADATA_PATH = (
    PROJECT_ROOT / "ml" / "experiments" / "graphsage_link_prediction" / "dataset_metadata.json"
)
DEFAULT_JSON = PROJECT_ROOT / "ml" / "reports" / "category_quality_report.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "ml" / "reports" / "category_quality_report.md"
FIELDS = {"country": "country", "industry": "industry"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_dataset_version() -> str:
    if not DATASET_METADATA_PATH.exists():
        return "unknown"
    return json.loads(DATASET_METADATA_PATH.read_text()).get("version", "unknown")


def fmt_int(value: Any) -> str:
    return f"{int(value):,}"


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def markdown_value_table(entries: list[dict[str, Any]], limit: int = 20) -> str:
    if not entries:
        return "_None._\n"
    lines = ["| Label | Raw value | Source | Nodes | Share of type |", "|---|---|---|---:|---:|"]
    for entry in entries[:limit]:
        lines.append(
            f"| {entry['normalized_value']} | `{entry['raw_value']}` | {source_label(entry['source'])} "
            f"| {fmt_int(entry['count'])} | {fmt_pct(entry['share_of_type'])} |"
        )
    return "\n".join(lines) + "\n"


def build_validation(
    frame: pd.DataFrame,
    field: str,
    result: dict[str, Any],
    raw_value_counts: Counter[str],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    counts = result["counts"]
    checks.append(
        {
            "check": f"{field}: category-type counts sum to total nodes",
            "expected": result["total_nodes"],
            "actual": sum(counts[category_type] for category_type in ("verified", "anonymized", "missing", "unknown")),
        }
    )
    source_totals = frame["source_dataset"].value_counts().to_dict()
    checks.append(
        {
            "check": f"{field}: per-source totals match processed nodes.csv",
            "expected": {str(key): int(value) for key, value in source_totals.items()},
            "actual": {source: item["total"] for source, item in result["by_source"].items()},
        }
    )
    nonmissing = result["total_nodes"] - counts["missing"]
    checks.append(
        {
            "check": f"{field}: raw distinct-value counts sum to non-missing nodes",
            "expected": int(nonmissing),
            "actual": sum(item["count"] for item in result["raw_distinct_values"]),
        }
    )
    distinct_expected = len(raw_value_counts)
    distinct_actual = len(result["raw_distinct_values"])
    checks.append(
        {
            "check": f"{field}: distinct raw values match direct pandas value_counts",
            "expected": int(distinct_expected),
            "actual": int(distinct_actual),
        }
    )
    sampled = [entry for entry in result["top_verified"] + result["top_anonymized"] if entry["raw_value"]]
    checks.append(
        {
            "check": f"{field}: top displayed values match direct raw counts",
            "expected": {entry["raw_value"]: raw_value_counts[entry["raw_value"]] for entry in sampled},
            "actual": {entry["raw_value"]: entry["count"] for entry in sampled},
        }
    )
    for check in checks:
        check["passed"] = check["expected"] == check["actual"]
    return checks


def build_report() -> dict[str, Any]:
    frame = pd.read_csv(NODES_PATH, dtype=str, keep_default_na=False)
    if "source_dataset" not in frame.columns:
        raise ValueError("nodes.csv must contain source_dataset")
    reports: dict[str, Any] = {}
    validation: list[dict[str, Any]] = []
    for field, column in FIELDS.items():
        if column not in frame.columns:
            raise ValueError(f"nodes.csv must contain {column}")
        result = aggregate_categories(
            frame[column].tolist(),
            frame["source_dataset"].tolist(),
            field,
            top_n=25,
            include_all_values=True,
        )
        reports[field] = result
        raw_counts = Counter(
            str(value).strip()
            for value in frame[column].tolist()
            if str(value).strip()
        )
        validation.extend(build_validation(frame, field, result, raw_counts))
    before_country = (
        frame.loc[frame["country"].str.strip().ne(""), "country"].str.strip().value_counts().head(12).to_dict()
    )
    before_industry = (
        frame.loc[frame["industry"].str.strip().ne(""), "industry"].str.strip().value_counts().head(12).to_dict()
    )
    return {
        "generated_at": utc_now(),
        "dataset_version": load_dataset_version(),
        "input": str(NODES_PATH.relative_to(PROJECT_ROOT)),
        "total_nodes": int(len(frame)),
        "root_cause": {
            "source_file": "datasets/raw/ic_splc/extracted_csv/ic_splc_nodes.csv",
            "columns": ["country", "industry_1st", "category_1st"],
            "explanation": (
                "IC-SPLC categorical values are anonymized frequency-ranked codes. "
                "The publisher does not provide a mapping to real countries or industries. "
                "preprocess_ic_splc.py preserves them, build_graph_dataset.py combines them with "
                "AnonNet/SAMPOQ nodes, and the dashboard previously aggregated them without source context."
            ),
            "metadata": "datasets/raw/ic_splc/mendeley_metadata.json",
        },
        "before_fix": {
            "country_top_mixed": {str(key): int(value) for key, value in before_country.items()},
            "industry_top_codes": {str(key): int(value) for key, value in before_industry.items()},
        },
        "fields": reports,
        "validation": {
            "passed": all(check["passed"] for check in validation),
            "checks": validation,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    country = report["fields"]["country"]
    industry = report["fields"]["industry"]
    lines = [
        "# Country and Industry Category-Quality Report",
        "",
        f"Generated: `{report['generated_at']}`  ",
        f"Dataset version: `{report['dataset_version']}`  ",
        f"Input: `{report['input']}`  ",
        f"Total nodes: **{fmt_int(report['total_nodes'])}**",
        "",
        "## Root cause",
        "",
        f"- Source file: `{report['root_cause']['source_file']}`",
        f"- Source columns: {', '.join(f'`{column}`' for column in report['root_cause']['columns'])}",
        f"- {report['root_cause']['explanation']}",
        f"- Source metadata: `{report['root_cause']['metadata']}`",
        "",
        "## Summary",
        "",
        "| Field | Verified nodes | Anonymized nodes | Missing nodes | Unknown nodes | Verified distinct | Anonymized distinct |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for field, item in (("Country", country), ("Industry", industry)):
        counts = item["counts"]
        distinct = item["distinct"]
        lines.append(
            f"| {field} | {fmt_int(counts['verified'])} | {fmt_int(counts['anonymized'])} "
            f"| {fmt_int(counts['missing'])} | {fmt_int(counts['unknown'])} "
            f"| {fmt_int(distinct['verified'])} | {fmt_int(distinct['anonymized'])} |"
        )
    lines.extend(
        [
            "",
            "## Dataset-wise category distribution",
            "",
            "| Source | Total | Country verified | Country anonymized | Country missing | Industry verified | Industry anonymized | Industry missing |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    sources = sorted(set(country["by_source"]) | set(industry["by_source"]))
    for source in sources:
        c = country["by_source"].get(source, {})
        i = industry["by_source"].get(source, {})
        lines.append(
            f"| {source_label(source)} | {fmt_int(c.get('total', 0))} | {fmt_int(c.get('verified', 0))} "
            f"| {fmt_int(c.get('anonymized', 0))} | {fmt_int(c.get('missing', 0))} "
            f"| {fmt_int(i.get('verified', 0))} | {fmt_int(i.get('anonymized', 0))} "
            f"| {fmt_int(i.get('missing', 0))} |"
        )
    lines.extend(
        [
            "",
            "## Verified countries",
            "",
            "Source: AnonNet `location`. Names are preserved exactly as provided.",
            "",
            markdown_value_table(country["top_verified"]),
            "## Anonymized country codes",
            "",
            "Source: IC-SPLC `country`. These are anonymized codes, not countries.",
            "",
            markdown_value_table(country["top_anonymized"]),
            "## Verified industries",
            "",
            "No source in the current graph provides verified industry names.",
            "",
            markdown_value_table(industry["top_verified"]),
            "## Anonymized industry codes",
            "",
            "Source: IC-SPLC `industry_1st`. These are anonymized codes, not industries.",
            "",
            markdown_value_table(industry["top_anonymized"]),
            "## Missing values by source",
            "",
        ]
    )
    for field, item in (("Country", country), ("Industry", industry)):
        parts = ", ".join(
            f"{source_label(source)}: {fmt_int(count)}"
            for source, count in item["missing_by_source"].items()
        )
        lines.append(f"- **{field}:** {parts or 'none'}")
    lines.extend(
        [
            "",
            "## Before the fix",
            "",
            "The previous dashboard aggregated all non-empty raw values without source context.",
            "",
            "| Rank | Mixed top countries | Count | Raw top industries | Count |",
            "|---:|---|---:|---|---:|",
        ]
    )
    before_country = list(report["before_fix"]["country_top_mixed"].items())
    before_industry = list(report["before_fix"]["industry_top_codes"].items())
    for index in range(max(len(before_country), len(before_industry))):
        country_value, country_count = before_country[index] if index < len(before_country) else ("", "")
        industry_value, industry_count = before_industry[index] if index < len(before_industry) else ("", "")
        lines.append(f"| {index + 1} | `{country_value}` | {fmt_int(country_count) if country_count != '' else ''} | `{industry_value}` | {fmt_int(industry_count) if industry_count != '' else ''} |")
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"Overall status: **{'PASS' if report['validation']['passed'] else 'FAIL'}**",
            "",
            "| Check | Status | Expected | Actual |",
            "|---|---|---|---|",
        ]
    )
    for check in report["validation"]["checks"]:
        lines.append(
            f"| {check['check']} | {'PASS' if check['passed'] else 'FAIL'} | "
            f"`{check['expected']}` | `{check['actual']}` |"
        )
    lines.extend(
        [
            "",
            "Full raw and normalized distinct-value lists are in `category_quality_report.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate category-quality reports")
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    args = parser.parse_args()
    report = build_report()
    json_path = Path(args.json)
    markdown_path = Path(args.markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    if not report["validation"]["passed"]:
        print("Validation failed; see report.")
        return 1
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), "validation": "PASS"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
