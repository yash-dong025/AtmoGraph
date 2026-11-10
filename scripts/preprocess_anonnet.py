from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    PROCESSED_DIR,
    RAW_DIR,
    clean_text,
    ensure_dirs,
    setup_logging,
    stable_id,
    write_csv,
)

RAW = RAW_DIR / "anonnet" / "repo" / "mim2025_network_challenge"
NODES = RAW / "nodes_anon.csv"
EDGES = RAW / "edges_anon.csv"
OUT_COMPANIES = PROCESSED_DIR / "companies" / "companies_anonnet.csv"
OUT_EDGES = PROCESSED_DIR / "relationships" / "supply_chain_edges_anonnet.csv"

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
    "products",
    "product_count",
    "name_is_anonymised",
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess AnonNet anonymised supply network")
    args = parser.parse_args()
    ensure_dirs()
    logger = setup_logging("preprocess_anonnet")
    if not NODES.exists() or not EDGES.exists():
        raise SystemExit(f"AnonNet files missing under {RAW}")
    nodes = pd.read_csv(NODES, dtype=str).fillna("")
    edges = pd.read_csv(EDGES, dtype=str).fillna("")
    logger.info("Loaded %d nodes and %d edges", len(nodes), len(edges))
    node_map = {r["id"]: r for _, r in nodes.iterrows()}

    companies = []
    for _, r in nodes.iterrows():
        products = clean_text(r.get("products"))
        plist = [p.strip() for p in (products or "").split(",") if p.strip()]
        companies.append(
            {
                "company_id": f"anonnet:{r['id']}",
                "company_name": f"AnonNet entity {r['id']}",
                "original_company_name": None,
                "ticker": None,
                "country": clean_text(r.get("location")),
                "industry": None,
                "product_category": plist[0] if plist else None,
                "source_dataset": "anonnet",
                "source_record_id": r["id"],
                "products": products,
                "product_count": len(plist) if plist else None,
                "name_is_anonymised": True,
                "id_method": "source_anonymised_id",
            }
        )

    edge_rows = []
    dropped_self = 0
    for _, r in edges.iterrows():
        s = r["source"]
        t = r["target"]
        if s == t:
            dropped_self += 1
            continue
        src_products = clean_text(node_map.get(s, {}).get("products"))
        first_product = None
        if src_products:
            first_product = [p.strip() for p in src_products.split(",") if p.strip()][0]
        edge_rows.append(
            {
                "edge_id": stable_id("anonnet_edge", f"{s}|{t}"),
                "source_company_id": f"anonnet:{s}",
                "target_company_id": f"anonnet:{t}",
                "relationship_type": "supply_relationship_anonymised",
                "product_category": first_product,
                "year": None,
                "filing_date": None,
                "source_dataset": "anonnet",
                "source_record_id": f"{s}-{t}",
                "confidence": 0.8,
                "relationship_direction": "As provided (source -> target); likely Supplier -> Customer",
            }
        )
    n_comp = write_csv(OUT_COMPANIES, companies, COMPANY_FIELDS)
    n_edge = write_csv(OUT_EDGES, edge_rows, EDGE_FIELDS)
    stats = {
        "nodes": n_comp,
        "edges": n_edge,
        "input_edges": len(edges),
        "dropped_self_loop": dropped_self,
        "missing_country": int(nodes["location"].eq("").sum()),
        "missing_products": int(nodes["products"].eq("").sum()),
        "unique_countries": int(nodes["location"].nunique()),
        "unique_products": int(nodes["products"].nunique()),
        "direction_note": "Edge direction is preserved exactly as provided; the source does not publish a formal direction definition.",
        "merge_warning": "Anonymised IDs must not be merged with identifiable company records.",
    }
    logger.info("AnonNet processed: %d companies, %d edges", n_comp, n_edge)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
