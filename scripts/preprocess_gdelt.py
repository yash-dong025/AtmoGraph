from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    PROCESSED_DIR,
    RAW_DIR,
    clean_text,
    ensure_dirs,
    read_json,
    setup_logging,
    stable_id,
    utc_now,
    write_csv,
    write_json,
)

GDELT = RAW_DIR / "gdelt"
API_DIR = GDELT / "api"
GKG_FILTERED = GDELT / "gkg_filtered"
COLLECTION_META = GDELT / "collection_metadata.json"

OUT_NEWS = PROCESSED_DIR / "disruptions" / "news_articles.csv"
OUT_EVENTS = PROCESSED_DIR / "disruptions" / "disruption_events.csv"

NEWS_FIELDS = [
    "article_id",
    "title",
    "url",
    "domain",
    "seendate",
    "language",
    "source_country",
    "query_keywords",
    "collection_method",
    "source_dataset",
    "collected_at",
]

EVENT_FIELDS = [
    "disruption_id",
    "event_title",
    "event_description",
    "disruption_type",
    "affected_company",
    "affected_country",
    "affected_region",
    "event_date",
    "source_url",
    "source_dataset",
    "extraction_method",
    "confidence",
    "verification_status",
    "publication_country",
    "mentioned_locations",
    "matched_keywords",
    "article_id",
]


def parse_seendate(value: str) -> str | None:
    s = clean_text(value) or ""
    if re.fullmatch(r"\d{8}T\d{6}Z", s):
        return datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    if re.fullmatch(r"\d{14}", s):
        return datetime.strptime(s, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    return s or None


def keyword_to_type(kw: str) -> str:
    mapping = {
        "port strike": "port_strike",
        "port congestion": "port_congestion",
        "factory shutdown": "factory_shutdown",
        "supply chain disruption": "supply_chain_disruption",
        "shipping delay": "shipping_delay",
        "logistics disruption": "logistics_disruption",
        "natural disaster supply chain": "natural_disaster",
        "supplier disruption": "supplier_disruption",
    }
    return mapping.get(kw.lower(), "supply_chain_disruption")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess GDELT news metadata")
    args = parser.parse_args()
    ensure_dirs()
    logger = setup_logging("preprocess_gdelt")
    meta = read_json(COLLECTION_META, default={})
    collected_at = meta.get("finished_at") or utc_now()

    articles: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    query_counts: Counter[str] = Counter()

    for path in sorted(API_DIR.glob("*.json")) if API_DIR.exists() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            logger.warning("Skipping unreadable GDELT API file %s: %s", path.name, exc)
            continue
        for art in payload.get("articles", []):
            url = clean_text(art.get("url"))
            if not url:
                continue
            kw = clean_text(payload.get("query")) or path.stem.replace("_", " ")
            query_counts[kw] += 1
            aid = stable_id("gdelt", url, 20)
            row = articles.setdefault(
                aid,
                {
                    "article_id": aid,
                    "title": clean_text(art.get("title")),
                    "url": url,
                    "domain": clean_text(art.get("domain")),
                    "seendate": parse_seendate(art.get("seendate")),
                    "language": clean_text(art.get("language")),
                    "source_country": clean_text(art.get("sourcecountry")),
                    "query_keywords": set(),
                    "collection_method": "gdelt_doc_api",
                    "source_dataset": "gdelt",
                    "collected_at": collected_at,
                },
            )
            row["query_keywords"].add(kw)
            events.append(
                {
                    "disruption_id": stable_id("gdelt_event", f"{aid}|{kw}", 20),
                    "event_title": clean_text(art.get("title")),
                    "event_description": None,
                    "disruption_type": keyword_to_type(kw),
                    "affected_company": None,
                    "affected_country": None,
                    "affected_region": None,
                    "event_date": (row["seendate"] or "")[:10] or None,
                    "source_url": url,
                    "source_dataset": "gdelt",
                    "extraction_method": "doc_api_keyword_query",
                    "confidence": 0.4,
                    "verification_status": "unverified_news_mention",
                    "publication_country": clean_text(art.get("sourcecountry")),
                    "mentioned_locations": None,
                    "matched_keywords": kw,
                    "article_id": aid,
                }
            )

    for path in sorted(GKG_FILTERED.glob("*.csv")) if GKG_FILTERED.exists() else []:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                url = clean_text(r.get("url"))
                if not url:
                    continue
                aid = stable_id("gdelt", url, 20)
                kw = clean_text(r.get("matched_keyword")) or ""
                dtype = clean_text(r.get("disruption_type")) or keyword_to_type(kw)
                match_source = clean_text(r.get("match_source")) or ""
                row = articles.setdefault(
                    aid,
                    {
                        "article_id": aid,
                        "title": clean_text(r.get("title")),
                        "url": url,
                        "domain": clean_text(r.get("domain")),
                        "seendate": parse_seendate(r.get("timestamp")),
                        "language": None,
                        "source_country": None,
                        "query_keywords": set(),
                        "collection_method": "gdelt_gkg_sample",
                        "source_dataset": "gdelt",
                        "collected_at": collected_at,
                    },
                )
                row["query_keywords"].add(kw)
                confidence = 0.35 if match_source == "keyword_phrase" else 0.2
                events.append(
                    {
                        "disruption_id": stable_id("gdelt_event", f"{aid}|{kw}|{match_source}", 20),
                        "event_title": clean_text(r.get("title")),
                        "event_description": None,
                        "disruption_type": dtype,
                        "affected_company": None,
                        "affected_country": None,
                        "affected_region": None,
                        "event_date": (row["seendate"] or "")[:10] or None,
                        "source_url": url,
                        "source_dataset": "gdelt",
                        "extraction_method": "gkg_keyword_theme_rule",
                        "confidence": confidence,
                        "verification_status": "unverified_news_mention",
                        "publication_country": None,
                        "mentioned_locations": clean_text(r.get("locations")),
                        "matched_keywords": kw,
                        "article_id": aid,
                    }
                )

    news_rows = []
    for row in articles.values():
        row = dict(row)
        row["query_keywords"] = ";".join(sorted(row["query_keywords"]))
        news_rows.append(row)
    n_news = write_csv(OUT_NEWS, news_rows, NEWS_FIELDS)
    n_events = write_csv(OUT_EVENTS, events, EVENT_FIELDS)

    dates = sorted(r["seendate"][:10] for r in news_rows if r.get("seendate"))
    stats = {
        "news_articles": n_news,
        "disruption_events": n_events,
        "unique_domains": len({r["domain"] for r in news_rows if r.get("domain")}),
        "date_range": [dates[0], dates[-1]] if dates else None,
        "collection_methods": dict(Counter(r["collection_method"] for r in news_rows)),
        "disruption_type_counts": dict(Counter(e["disruption_type"] for e in events)),
        "query_counts": dict(query_counts),
        "verification_note": "All GDELT records are unverified news mentions, not confirmed disruption events or shipment delays.",
    }
    write_json(PROCESSED_DIR / "disruptions" / "gdelt_preprocessing_stats.json", stats)
    logger.info("GDELT processed: %d articles, %d event mentions", n_news, n_events)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
