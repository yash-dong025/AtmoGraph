from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    DATASETS_DIR,
    EVALUATION_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    clean_text,
    ensure_dirs,
    normalise_name,
    setup_logging,
    slugify,
    stable_id,
    utc_now,
    write_csv,
    write_json,
)

REPO = RAW_DIR / "disruption_monitoring" / "restricted" / "repo"
KG = REPO / "dataset" / "supplychainKG_clean.csv"
OUT_DIR = PROCESSED_DIR / "restricted" / "disruption_monitoring"
OUT_COMPANIES = OUT_DIR / "companies.csv"
OUT_EDGES = OUT_DIR / "edges.csv"

COMPANY_FIELDS = [
    "company_id",
    "company_name",
    "original_company_name",
    "ticker",
    "country",
    "industry",
    "product_category",
    "source_dataset",
    "source_record_id",
    "id_method",
]

EDGE_FIELDS = [
    "edge_id",
    "source_company_id",
    "target_company_id",
    "relationship_type",
    "product_category",
    "year",
    "filing_date",
    "source_dataset",
    "source_record_id",
    "confidence",
    "relationship_direction",
]

NOTICE = """# RESTRICTED DATA - CC BY-NC 4.0

This folder contains material from:
https://github.com/sara-almahri/supply-chain-disruption-monitoring

Licence: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
Copyright (c) 2025-2026 Sara AlMahri.

NON-COMMERCIAL USE ONLY. Do not include these files in a company deliverable,
commercial deployment, or a product offering unless the company has confirmed
that the intended use is permitted or has obtained separate permission.

This material is kept outside the main processed graph and training datasets.
"""


def copy_eval_files(logger) -> dict[str, Any]:
    scenarios_src = REPO / "evaluation" / "scenarios"
    gt_src = REPO / "evaluation" / "ground_truth_final"
    benchmark_src = REPO / "evaluation" / "benchmark_v2"
    scen_dst = EVALUATION_DIR / "scenarios" / "restricted"
    gt_dst = EVALUATION_DIR / "ground_truth" / "restricted"
    bench_dst = EVALUATION_DIR / "benchmark" / "restricted"
    for d in [scen_dst, gt_dst, bench_dst]:
        d.mkdir(parents=True, exist_ok=True)
        (d / "RESTRICTED_NOTICE.md").write_text(NOTICE, encoding="utf-8")
    copied = {"scenarios": 0, "ground_truth": 0, "benchmark": 0}
    if scenarios_src.exists():
        for p in sorted(scenarios_src.glob("*.json")):
            shutil.copy2(p, scen_dst / p.name)
            copied["scenarios"] += 1
    if gt_src.exists():
        for p in sorted(gt_src.glob("*.json")):
            shutil.copy2(p, gt_dst / p.name)
            copied["ground_truth"] += 1
    for rel in [
        "items/test/test_set.json",
        "items/test/specs.json",
        "items/test/ground_truth_summary.json",
        "items/test/generation_summary.json",
    ]:
        p = benchmark_src / rel
        if p.exists():
            target = bench_dst / Path(rel).name
            shutil.copy2(p, target)
            copied["benchmark"] += 1
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess restricted CC BY-NC disruption monitoring data")
    parser.add_argument("--allow-restricted", action="store_true", help="Explicitly allow processing restricted data into datasets/processed/restricted")
    args = parser.parse_args()
    ensure_dirs()
    logger = setup_logging("preprocess_disruption_monitoring")
    if not args.allow_restricted:
        print(json.dumps({"status": "skipped_license", "reason": "Pass --allow-restricted after confirming permission. CC BY-NC 4.0 is non-commercial."}, indent=2))
        return 0
    if not KG.exists():
        raise SystemExit(f"Restricted KG not found: {KG}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(KG, dtype=str).fillna("")
    logger.info("Loaded %d restricted KG rows", len(df))

    companies: dict[str, dict[str, Any]] = {}

    def add(name, industry, country, record_id):
        key = normalise_name(name)
        cid = f"disruption:{slugify(name, 40)}_{stable_id('disruption_name', key, 8)}"
        if cid not in companies:
            companies[cid] = {
                "company_id": cid,
                "company_name": clean_text(name),
                "original_company_name": clean_text(name),
                "ticker": None,
                "country": clean_text(country),
                "industry": clean_text(industry),
                "product_category": None,
                "source_dataset": "disruption_monitoring_restricted",
                "source_record_id": record_id,
                "id_method": "normalised_name_hash",
            }
        else:
            if not companies[cid]["country"] and country:
                companies[cid]["country"] = clean_text(country)
            if not companies[cid]["industry"] and industry:
                companies[cid]["industry"] = clean_text(industry)
        return cid

    edges = []
    for idx, r in df.iterrows():
        sid = add(r.get("Supplier"), r.get("Supplier Industry"), r.get("Supplier Country"), f"row:{idx}:supplier")
        tid = add(r.get("Customer"), r.get("Customer Industry"), r.get("Customer Country"), f"row:{idx}:customer")
        if sid == tid:
            continue
        edges.append(
            {
                "edge_id": stable_id("disruption_edge", f"{sid}|{tid}|{idx}"),
                "source_company_id": sid,
                "target_company_id": tid,
                "relationship_type": "supplies_to",
                "product_category": None,
                "year": None,
                "filing_date": None,
                "source_dataset": "disruption_monitoring_restricted",
                "source_record_id": str(idx),
                "confidence": 0.8,
                "relationship_direction": "Supplier -> Customer",
            }
        )
    n_comp = write_csv(OUT_COMPANIES, companies.values(), COMPANY_FIELDS)
    n_edge = write_csv(OUT_EDGES, edges, EDGE_FIELDS)
    copied = copy_eval_files(logger)
    scenarios = json.loads((REPO / "evaluation" / "scenarios" / "multi_company_scenarios.json").read_text(encoding="utf-8"))
    gt_index = json.loads((REPO / "evaluation" / "ground_truth_final" / "index.json").read_text(encoding="utf-8"))
    test_set = json.loads((REPO / "evaluation" / "benchmark_v2" / "items" / "test" / "test_set.json").read_text(encoding="utf-8"))
    meta = {
        "generated_at": utc_now(),
        "restricted": True,
        "license": "CC BY-NC 4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
        "copyright": "Copyright (c) 2025-2026 Sara AlMahri",
        "source_url": "https://github.com/sara-almahri/supply-chain-disruption-monitoring",
        "commercial_use": "NOT PERMITTED without separate approval",
        "included_in_main_graph": False,
        "included_in_main_training": False,
        "kg_rows": len(df),
        "companies": n_comp,
        "edges": n_edge,
        "scenario_count": len(scenarios),
        "scenario_ids": [s.get("scenario_id") for s in scenarios],
        "ground_truth_files": gt_index.get("scenarios", []),
        "benchmark_items": len(test_set.get("items", [])),
        "benchmark_description": test_set.get("description"),
        "scenario_note": "Scenarios are synthesised news articles with ground-truth references generated by the repository authors. They are not real observed disruptions.",
        "copied_files": copied,
    }
    write_json(EVALUATION_DIR / "evaluation_metadata.json", meta)
    write_json(OUT_DIR / "preprocessing_stats.json", meta)
    logger.info("Restricted KG processed: %d companies, %d edges; evaluation copies: %s", n_comp, n_edge, copied)
    print(json.dumps({k: meta[k] for k in ["companies", "edges", "scenario_count", "benchmark_items", "copied_files"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
