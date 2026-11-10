from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common import (
    DATASETS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    clean_text,
    ensure_dirs,
    rel,
    setup_logging,
    stable_id,
    utc_now,
    write_csv,
    write_json,
)

RAW = RAW_DIR / "ic_splc"
EXTRACT_DIR = RAW / "extracted_csv"
NODES_CSV = EXTRACT_DIR / "ic_splc_nodes.csv"
EDGES_CSV = EXTRACT_DIR / "ic_splc_edges.csv"
EMB_PARQUET = EXTRACT_DIR / "ic_splc_embeddings.parquet"
EXTRACT_META = EXTRACT_DIR / "extraction_metadata.json"

OUT_COMPANIES = PROCESSED_DIR / "companies" / "companies_ic_splc.csv"
OUT_EDGES = PROCESSED_DIR / "relationships" / "supply_chain_edges_ic_splc.csv"

NODE_FIELDS = [
    "id",
    "country",
    "industry_1st",
    "industry_2nd",
    "category_1st",
    "category_2nd",
    "category_3rd",
    "other_possible_industry_2",
    "other_possible_category_3rd",
    "selected",
    "embedding_available",
]

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
    "industry_2nd",
    "category_2nd",
    "category_3rd",
    "other_possible_industry_2",
    "other_possible_category_3rd",
    "embedding_available",
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


def find_dump() -> Path | None:
    for p in RAW.rglob("*.dump"):
        return p
    return None


def extract_from_neo4j(uri: str, logger) -> dict[str, Any]:
    try:
        from neo4j import GraphDatabase
    except Exception as exc:
        raise RuntimeError("neo4j Python driver is required for extraction: pip install neo4j") from exc
    dump = find_dump()
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    driver = GraphDatabase.driver(uri, auth=None)
    node_query = """
        MATCH (n:EntityObj)
        RETURN id(n) AS id,
               n.country AS country,
               n.industry_1st AS industry_1st,
               n.industry_2nd AS industry_2nd,
               n.category_1st AS category_1st,
               n.category_2nd AS category_2nd,
               n.category_3rd AS category_3rd,
               n.other_possible_industry_2 AS other_possible_industry_2,
               n.other_possible_category_3rd AS other_possible_category_3rd,
               n.selected AS selected,
               n.embedding AS embedding
        ORDER BY id(n)
    """
    edge_query = """
        MATCH (a:EntityObj)-[r:SupplyProductTo]->(b:EntityObj)
        RETURN id(a) AS source_id, id(b) AS target_id, r.year AS year
        ORDER BY source_id, target_id, year
    """
    with driver.session() as session:
        node_records = list(session.run(node_query))
        edge_records = list(session.run(edge_query))
        version = session.run("CALL dbms.components() YIELD versions RETURN versions[0] AS v").single()["v"]
    logger.info("Extracted %d nodes and %d relationships from Neo4j", len(node_records), len(edge_records))
    emb_rows = []
    with open(NODES_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=NODE_FIELDS)
        w.writeheader()
        for rec in node_records:
            emb = rec.get("embedding")
            row = {k: rec.get(k) for k in NODE_FIELDS if k != "embedding_available"}
            row["embedding_available"] = 1 if emb else 0
            for k in ["other_possible_industry_2", "other_possible_category_3rd"]:
                if isinstance(row.get(k), list):
                    row[k] = ";".join(str(x) for x in row[k])
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in NODE_FIELDS})
            if emb:
                er = {"node_id": int(rec["id"])}
                for i, v in enumerate(emb):
                    er[f"emb_{i}"] = float(v)
                emb_rows.append(er)
    with open(EDGES_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["source_id", "target_id", "year"])
        w.writeheader()
        for rec in edge_records:
            w.writerow({"source_id": rec["source_id"], "target_id": rec["target_id"], "year": rec["year"]})
    if emb_rows:
        import pyarrow as pa
        import pyarrow.parquet as pq

        cols = list(emb_rows[0].keys())
        table = pa.table({c: [r.get(c) for r in emb_rows] for c in cols})
        pq.write_table(table, EMB_PARQUET)
    meta = {
        "extracted_at": utc_now(),
        "neo4j_uri": uri,
        "neo4j_version": version,
        "dump_file": rel(dump) if dump else None,
        "dump_sha256": None,
        "nodes": len(node_records),
        "relationships": len(edge_records),
        "embeddings_written": len(emb_rows),
        "node_query": " ".join(node_query.split()),
        "edge_query": " ".join(edge_query.split()),
        "direction_note": "SupplyProductTo edges are directed supplier -> customer as documented by the dataset authors.",
    }
    if dump:
        from common import sha256_file

        meta["dump_sha256"] = sha256_file(dump)
    write_json(EXTRACT_META, meta)
    driver.close()
    return meta


def process(logger) -> dict[str, Any]:
    if not NODES_CSV.exists() or not EDGES_CSV.exists():
        raise RuntimeError("IC-SPLC extracted CSVs are missing. Run with --extract and a running Neo4j instance.")
    with open(NODES_CSV, "r", encoding="utf-8", newline="") as fh:
        node_rows = list(csv.DictReader(fh))
    with open(EDGES_CSV, "r", encoding="utf-8", newline="") as fh:
        edge_rows = list(csv.DictReader(fh))
    node_ids = {int(r["id"]) for r in node_rows}
    companies = []
    for r in node_rows:
        nid = int(r["id"])
        companies.append(
            {
                "company_id": f"icsplc:{nid}",
                "company_name": f"IC-SPLC entity {nid}",
                "original_company_name": None,
                "ticker": None,
                "country": clean_text(r.get("country")),
                "industry": clean_text(r.get("industry_1st")),
                "product_category": clean_text(r.get("category_1st")),
                "source_dataset": "ic_splc",
                "source_record_id": str(nid),
                "industry_2nd": clean_text(r.get("industry_2nd")),
                "category_2nd": clean_text(r.get("category_2nd")),
                "category_3rd": clean_text(r.get("category_3rd")),
                "other_possible_industry_2": clean_text(r.get("other_possible_industry_2")),
                "other_possible_category_3rd": clean_text(r.get("other_possible_category_3rd")),
                "embedding_available": int(r.get("embedding_available") or 0),
                "name_is_anonymised": True,
                "id_method": "neo4j_internal_id",
            }
        )
    node_meta = {int(r["id"]): r for r in node_rows}
    edges = []
    dropped_missing = 0
    dropped_self = 0
    for r in edge_rows:
        s = int(r["source_id"])
        t = int(r["target_id"])
        if s not in node_ids or t not in node_ids:
            dropped_missing += 1
            continue
        if s == t:
            dropped_self += 1
            continue
        year = clean_text(r.get("year"))
        src_cat = clean_text(node_meta[s].get("category_1st"))
        edges.append(
            {
                "edge_id": stable_id("icsplc_edge", f"{s}|{t}|{year}"),
                "source_company_id": f"icsplc:{s}",
                "target_company_id": f"icsplc:{t}",
                "relationship_type": "supplies_product_to",
                "product_category": src_cat,
                "year": year,
                "filing_date": None,
                "source_dataset": "ic_splc",
                "source_record_id": f"{s}-{t}-{year}",
                "confidence": 0.7,
                "relationship_direction": "Supplier -> Customer",
            }
        )
    n_comp = write_csv(OUT_COMPANIES, companies, COMPANY_FIELDS)
    n_edge = write_csv(OUT_EDGES, edges, EDGE_FIELDS)
    logger.info("IC-SPLC processed: %d companies, %d edges", n_comp, n_edge)
    return {
        "companies": n_comp,
        "edges": n_edge,
        "dropped_missing_endpoint": dropped_missing,
        "dropped_self_loop": dropped_self,
        "input_nodes": len(node_rows),
        "input_edges": len(edge_rows),
        "embeddings_available": sum(1 for r in node_rows if str(r.get("embedding_available")) == "1"),
        "confidence_note": "0.7 reflects an LLM-extracted, anonymised source; not independently verified.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract and preprocess IC-SPLC-v2")
    parser.add_argument("--extract", action="store_true", help="Extract nodes/edges from a loaded Neo4j database")
    parser.add_argument("--uri", default="neo4j://127.0.0.1:7687")
    args = parser.parse_args()
    ensure_dirs()
    logger = setup_logging("preprocess_ic_splc")
    if args.extract:
        extract_from_neo4j(args.uri, logger)
    stats = process(logger)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
