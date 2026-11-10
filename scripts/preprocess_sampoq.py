from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    PROCESSED_DIR,
    RAW_DIR,
    clean_text,
    ensure_dirs,
    normalise_name,
    parse_date,
    setup_logging,
    slugify,
    stable_id,
    write_csv,
    write_json,
)

RAW = RAW_DIR / "sampoq" / "supply-chain.csv"
OUT_COMPANIES = PROCESSED_DIR / "companies" / "companies_sampoq.csv"
OUT_EDGES = PROCESSED_DIR / "relationships" / "supply_chain_edges_sampoq.csv"

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
    "cik",
    "id_method",
    "name_variants",
    "mention_count",
    "first_seen_filing_date",
    "last_seen_filing_date",
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
    "source_relation_type",
    "source_ticker",
    "related_ticker",
    "filing_accession",
    "duplicate_relationship",
]


def build_name_ticker_index(df: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        ticker = clean_text(row.get("source_ticker"))
        related_ticker = clean_text(row.get("related_ticker"))
        if ticker:
            mapping[normalise_name(row.get("source_name"))].add(ticker.upper())
        if related_ticker:
            mapping[normalise_name(row.get("related_name"))].add(related_ticker.upper())
    return {k: next(iter(v)) for k, v in mapping.items() if len(v) == 1 and k}


def resolve_entity(name: Any, ticker: Any, cik: Any, name_index: dict[str, str]) -> tuple[str, str]:
    name = clean_text(name)
    ticker = clean_text(ticker)
    cik = clean_text(cik)
    norm = normalise_name(name)
    if ticker:
        return f"sampoq:ticker:{ticker.upper()}", "ticker"
    if norm and norm in name_index:
        return f"sampoq:ticker:{name_index[norm]}", "name_resolved_to_ticker"
    if cik:
        return f"sampoq:cik:{cik}", "cik"
    if not norm:
        return f"sampoq:name:unknown_{stable_id('unknown', name)}", "unknown"
    return f"sampoq:name:{slugify(name, 48)}_{stable_id('name', norm, 8)}", "name"


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess SAMPOQ supply-chain 10-K relationships")
    parser.add_argument("--input", default=str(RAW))
    parser.add_argument("--output-companies", default=str(OUT_COMPANIES))
    parser.add_argument("--output-edges", default=str(OUT_EDGES))
    args = parser.parse_args()
    ensure_dirs()
    logger = setup_logging("preprocess_sampoq")
    path = Path(args.input)
    df = pd.read_csv(path, dtype=str).fillna("")
    for col in df.columns:
        df[col] = df[col].map(lambda x: clean_text(x) or "")
    logger.info("Loaded %d rows from %s", len(df), path)

    df["filing_date_iso"] = df["filing_date"].map(parse_date)
    df["year"] = df["filing_date_iso"].str[:4]
    name_index = build_name_ticker_index(df)

    entities: dict[str, dict[str, Any]] = {}
    mention_counter: Counter[str] = Counter()
    names_by_id: dict[str, Counter[str]] = defaultdict(Counter)
    tickers_by_id: dict[str, set[str]] = defaultdict(set)
    ciks_by_id: dict[str, set[str]] = defaultdict(set)
    dates_by_id: dict[str, list[str]] = defaultdict(list)
    methods: dict[str, str] = {}

    def add_entity(name, ticker, cik, date, record_id):
        cid, method = resolve_entity(name, ticker, cik, name_index)
        methods.setdefault(cid, method)
        mention_counter[cid] += 1
        if name:
            names_by_id[cid][name] += 1
        if ticker:
            tickers_by_id[cid].add(ticker.upper())
        if cik:
            ciks_by_id[cid].add(cik)
        if date:
            dates_by_id[cid].append(date)
        if cid not in entities:
            entities[cid] = {"source_record_id": record_id}
        return cid

    edges = []
    dropped_self = 0
    dropped_invalid = 0
    pair_counts: Counter[tuple[str, str, str]] = Counter()
    for idx, row in df.iterrows():
        date = row.get("filing_date_iso") or None
        record_id = f"{row.get('filing_accession','')}:{idx}"
        source_id = add_entity(row.get("source_name"), row.get("source_ticker"), row.get("source_cik"), date, record_id)
        related_id = add_entity(row.get("related_name"), row.get("related_ticker"), None, date, record_id)
        relation = (row.get("relation_type") or "").strip().lower()
        if source_id == related_id:
            dropped_self += 1
            continue
        if relation == "supplier":
            src, tgt = related_id, source_id
        elif relation == "customer":
            src, tgt = source_id, related_id
        else:
            dropped_invalid += 1
            continue
        pair_counts[(src, tgt, str(date))] += 1
        edges.append(
            {
                "edge_id": stable_id("sampoq_edge", f"{src}|{tgt}|{date}|{record_id}"),
                "source_company_id": src,
                "target_company_id": tgt,
                "relationship_type": "supplier_to_customer",
                "product_category": None,
                "year": row.get("year") or None,
                "filing_date": date,
                "source_dataset": "sampoq",
                "source_record_id": record_id,
                "confidence": 0.9,
                "relationship_direction": "Supplier -> Customer",
                "source_relation_type": relation,
                "source_ticker": row.get("source_ticker") or None,
                "related_ticker": row.get("related_ticker") or None,
                "filing_accession": row.get("filing_accession") or None,
                "duplicate_relationship": 1 if pair_counts[(src, tgt, str(date))] > 1 else 0,
            }
        )

    companies = []
    for cid, meta in entities.items():
        names = names_by_id[cid]
        display = names.most_common(1)[0][0] if names else None
        tickers = sorted(tickers_by_id[cid])
        ciks = sorted(ciks_by_id[cid])
        dates = sorted(d for d in dates_by_id[cid] if d)
        companies.append(
            {
                "company_id": cid,
                "company_name": display,
                "original_company_name": display,
                "ticker": tickers[0] if tickers else None,
                "country": None,
                "industry": None,
                "product_category": None,
                "source_dataset": "sampoq",
                "source_record_id": meta.get("source_record_id"),
                "cik": ciks[0] if ciks else None,
                "id_method": methods.get(cid),
                "name_variants": " | ".join(n for n, _ in names.most_common(10)),
                "mention_count": mention_counter[cid],
                "first_seen_filing_date": dates[0] if dates else None,
                "last_seen_filing_date": dates[-1] if dates else None,
            }
        )
    n_comp = write_csv(Path(args.output_companies), companies, COMPANY_FIELDS)
    n_edge = write_csv(Path(args.output_edges), edges, EDGE_FIELDS)
    dup_pairs = sum(1 for k, v in pair_counts.items() if v > 1)
    stats = {
        "input_rows": len(df),
        "companies": n_comp,
        "edges": n_edge,
        "dropped_self_loop": dropped_self,
        "dropped_unknown_relation": dropped_invalid,
        "duplicate_endpoint_date_pairs": dup_pairs,
        "date_range": [df["filing_date_iso"].min(), df["filing_date_iso"].max()],
        "relation_type_counts": df["relation_type"].value_counts().to_dict(),
        "direction_logic": {
            "supplier": "related_name is the filer's supplier: edge related -> filer",
            "customer": "related_name is the filer's customer: edge filer -> related",
            "output": "All edges are Supplier -> Customer.",
        },
        "entity_id_logic": "ticker > unambiguous name-to-ticker resolution > CIK > hashed normalised name",
        "confidence_note": "0.9 reflects source-reported relationships; not independently verified against SEC filings.",
    }
    write_json(PROCESSED_DIR / "companies" / "sampoq_preprocessing_stats.json", stats)
    logger.info("SAMPOQ processed: %d companies, %d edges", n_comp, n_edge)
    print(json.dumps(stats, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
