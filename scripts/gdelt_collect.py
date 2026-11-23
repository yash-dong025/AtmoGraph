from __future__ import annotations

import csv
import html
import io
import json
import re
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from common import RAW_DIR, human_bytes, sha256_file, utc_now, write_csv, write_json

GDELT_DIR = RAW_DIR / "gdelt"
API_DIR = GDELT_DIR / "api"
GKG_DIR = GDELT_DIR / "gkg"
GKG_FILTERED_DIR = GDELT_DIR / "gkg_filtered"
COLLECTION_META = GDELT_DIR / "collection_metadata.json"

DISRUPTION_KEYWORD_MAP = {
    "port strike": "port_strike",
    "port congestion": "port_congestion",
    "factory shutdown": "factory_shutdown",
    "supply chain disruption": "supply_chain_disruption",
    "shipping delay": "shipping_delay",
    "logistics disruption": "logistics_disruption",
    "natural disaster supply chain": "natural_disaster",
    "supplier disruption": "supplier_disruption",
}

THEME_HINTS = {
    "STRIKE": "labour_strike",
    "SHORTAGE": "supply_shortage",
    "CRISISLEX_C04_LOGISTICS_TRANSPORT": "logistics_disruption",
    "ECON_TRANSPORT_COST": "transport_cost_disruption",
    "WB_772_TRADE_FACILITATION_AND_LOGISTICS": "logistics_disruption",
    "WB_1803_TRANSPORT_INFRASTRUCTURE": "transport_infrastructure_disruption",
    "WB_164_MODES_OF_TRANSPORT": "transport_disruption",
}

GKG_COLUMNS = [
    "gkg_id",
    "timestamp",
    "source_collection_id",
    "domain",
    "url",
    "title",
    "themes",
    "locations",
    "organizations",
    "matched_keyword",
    "disruption_type",
    "match_source",
]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _extract_title(raw: str) -> str:
    m = re.search(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", raw or "", flags=re.S | re.I)
    if m:
        return html.unescape(re.sub(r"<[^>]+>", " ", m.group(1))).strip()
    return ""


def _gkg_match(title: str, themes: str, locations: str, keywords: list[str]) -> tuple[str, str, str]:
    text = f"{title} {themes} {locations}".lower()
    for kw in keywords:
        if kw.lower() in text:
            return kw, DISRUPTION_KEYWORD_MAP.get(kw.lower(), "supply_chain_disruption"), "keyword_phrase"
    for token, dtype in THEME_HINTS.items():
        if token.lower() in themes.lower():
            return token, dtype, "gkg_theme"
    return "", "", ""


def _request_with_retry(session: requests.Session, url: str, params: dict[str, Any] | None, cfg: dict[str, Any], logger, attempts: int | None = None):
    http_cfg = cfg.get("http", {})
    retries = attempts if attempts is not None else int(http_cfg.get("retries", 4))
    backoff = float(http_cfg.get("backoff_seconds", 5))
    last = None
    for i in range(retries):
        try:
            r = session.get(url, params=params, timeout=(int(http_cfg.get("connect_timeout", 20)), int(http_cfg.get("read_timeout", 180))))
            if r.status_code == 429:
                wait = backoff * (2 ** i)
                logger.warning("GDELT API rate limited (429); waiting %.0fs", wait)
                time.sleep(wait)
                last = r
                continue
            r.raise_for_status()
            return r
        except Exception as exc:
            last = exc
            wait = backoff * (2 ** i)
            logger.warning("GDELT request failed (%s); retrying in %.0fs", exc, wait)
            time.sleep(wait)
    if last is not None and hasattr(last, "status_code"):
        last.raise_for_status()
    raise RuntimeError(f"GDELT request failed after {retries} attempts: {last}")


def _api_window(date_range: dict[str, str]) -> tuple[str, str]:
    def fmt(s: str) -> str:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).strftime("%Y%m%d%H%M%S")

    return fmt(date_range["start"]), fmt(date_range["end"])


def collect_api(cfg: dict[str, Any], logger, force: bool = False) -> dict[str, Any]:
    API_DIR.mkdir(parents=True, exist_ok=True)
    gdelt = cfg["sources"]["gdelt"]
    session = requests.Session()
    session.headers.update({"User-Agent": cfg.get("http", {}).get("user_agent", "AtmoGraph-research/1.0")})
    start_dt, end_dt = _api_window(gdelt["date_range"])
    results = []
    for kw in gdelt["keywords"]:
        out = API_DIR / f"{_slug(kw)}.json"
        if out.exists() and not force:
            results.append({"keyword": kw, "status": "skipped_exists", "file": str(out)})
            continue
        params = {
            "query": kw,
            "mode": "artlist",
            "format": "json",
            "maxrecords": int(gdelt.get("api_max_records", 250)),
            "startdatetime": start_dt,
            "enddatetime": end_dt,
        }
        try:
            r = _request_with_retry(session, gdelt["api_url"], params, cfg, logger, attempts=int(gdelt.get("api_retries", 3)))
            payload = r.content
            out.write_bytes(payload)
            try:
                parsed = json.loads(payload.decode("utf-8", errors="replace"))
                n = len(parsed.get("articles", []))
            except Exception:
                n = 0
            results.append({"keyword": kw, "status": "ok", "file": str(out), "http_status": r.status_code, "articles": n})
        except Exception as exc:
            results.append({"keyword": kw, "status": "failed", "error": str(exc), "params": params})
            logger.error("GDELT API failed for %s: %s", kw, exc)
        time.sleep(float(gdelt.get("api_min_interval_seconds", 6)))
    ok = sum(1 for x in results if x["status"] in {"ok", "skipped_exists"})
    return {"method": "doc_api", "ok_queries": ok, "total_queries": len(results), "results": results}


def _candidate_timestamps(gdelt: dict[str, Any]) -> list[datetime]:
    start = datetime.fromisoformat(gdelt["date_range"]["start"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(gdelt["date_range"]["end"].replace("Z", "+00:00"))
    step = int(gdelt["gkg_fallback"].get("step_hours", 24))
    hour = int(gdelt["gkg_fallback"].get("snapshot_hour_utc", 4))
    max_files = int(gdelt["gkg_fallback"].get("max_files", 8))
    ts = []
    day = end
    while day > start and len(ts) < max_files:
        candidate = day.replace(hour=hour, minute=0, second=0, microsecond=0)
        if candidate > end:
            candidate -= timedelta(days=1)
        if candidate > start:
            ts.append(candidate)
        day -= timedelta(hours=step)
    return ts


def _find_available(session: requests.Session, base_url: str, ts: datetime, logger) -> str | None:
    for back in range(0, 8):
        t = ts - timedelta(minutes=15 * back)
        name = t.strftime("%Y%m%d%H%M%S") + ".gkg.csv.zip"
        url = f"{base_url}/{name}"
        try:
            r = session.head(url, timeout=30, allow_redirects=True)
            if r.status_code == 200:
                return url
        except Exception:
            continue
    return None


def _download_file(session: requests.Session, url: str, dest: Path, logger) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with session.get(url, stream=True, timeout=(20, 180)) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0) or 0)
            with open(part, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
        if total and part.stat().st_size != total:
            raise RuntimeError(f"size mismatch: got {part.stat().st_size}, expected {total}")
        part.replace(dest)
        logger.info("Downloaded %s (%s)", dest.name, human_bytes(dest.stat().st_size))
        return True
    except Exception as exc:
        logger.error("GKG download failed for %s: %s", url, exc)
        if part.exists():
            part.unlink(missing_ok=True)
        return False


def _filter_gkg_csv(zip_path: Path, out_path: Path, keywords: list[str], min_hits: int = 1) -> dict[str, Any]:
    rows = []
    scanned = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".gkg.csv")]
        if not names:
            return {"rows": 0, "scanned": 0}
        with zf.open(names[0]) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline="")
            for row in csv.reader(text, delimiter="\t"):
                scanned += 1
                if len(row) < 27:
                    continue
                title = _extract_title(row[26])
                kw, dtype, source = _gkg_match(title, row[7], row[9], keywords)
                if not kw:
                    continue
                rows.append(
                    {
                        "gkg_id": row[0],
                        "timestamp": row[1],
                        "source_collection_id": row[2],
                        "domain": row[3],
                        "url": row[4],
                        "title": title,
                        "themes": row[7],
                        "locations": row[9],
                        "organizations": row[13],
                        "matched_keyword": kw,
                        "disruption_type": dtype,
                        "match_source": source,
                    }
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_path, rows, GKG_COLUMNS)
    return {"rows": len(rows), "scanned": scanned}


def collect_gkg(cfg: dict[str, Any], logger, force: bool = False) -> dict[str, Any]:
    GKG_DIR.mkdir(parents=True, exist_ok=True)
    GKG_FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    gdelt = cfg["sources"]["gdelt"]
    fb = gdelt["gkg_fallback"]
    session = requests.Session()
    session.headers.update({"User-Agent": cfg.get("http", {}).get("user_agent", "AtmoGraph-research/1.0")})
    keywords = list(gdelt["keywords"])
    results = []
    for ts in _candidate_timestamps(gdelt):
        url = _find_available(session, fb["base_url"], ts, logger)
        if not url:
            results.append({"timestamp": ts.isoformat(), "status": "unavailable"})
            continue
        name = url.rsplit("/", 1)[-1]
        zip_path = GKG_DIR / name
        ok = _download_file(session, url, zip_path, logger) if (force or not zip_path.exists()) else True
        if not ok:
            results.append({"timestamp": ts.isoformat(), "status": "download_failed", "url": url})
            continue
        out_path = GKG_FILTERED_DIR / (name.replace(".gkg.csv.zip", ".filtered.csv"))
        if out_path.exists() and not force:
            with open(out_path, "r", encoding="utf-8") as fh:
                n = max(0, sum(1 for _ in fh) - 1)
            results.append({"timestamp": ts.isoformat(), "status": "skipped_exists", "url": url, "matched_rows": n})
            continue
        try:
            stats = _filter_gkg_csv(zip_path, out_path, keywords, int(fb.get("min_keyword_hits", 1)))
            results.append({"timestamp": ts.isoformat(), "status": "ok", "url": url, "zip": str(zip_path), "filtered": str(out_path), **stats})
        except Exception as exc:
            results.append({"timestamp": ts.isoformat(), "status": "filter_failed", "url": url, "error": str(exc)})
            logger.error("GKG filter failed for %s: %s", url, exc)
    ok = sum(1 for x in results if x["status"] in {"ok", "skipped_exists"})
    return {"method": "gkg_sample", "ok_files": ok, "total_files": len(results), "results": results}


def collect(cfg: dict[str, Any], logger, force: bool = False) -> dict[str, Any]:
    started = utc_now()
    api = collect_api(cfg, logger, force=force)
    meta: dict[str, Any] = {
        "started_at": started,
        "finished_at": None,
        "date_range": cfg["sources"]["gdelt"]["date_range"],
        "keywords": cfg["sources"]["gdelt"]["keywords"],
        "api": api,
        "fallback": None,
        "notes": [
            "GDELT DOC API is preferred. The API may return HTTP 429 for shared IPs.",
            "If the API is unavailable, a small reproducible sample of GDELT 2.0 GKG 15-minute files is filtered by the configured supply-chain keywords.",
            "GKG records are article metadata only. No full article text is downloaded.",
        ],
    }
    if api["ok_queries"] < api["total_queries"] and cfg["sources"]["gdelt"].get("gkg_fallback", {}).get("enabled", False):
        meta["fallback"] = collect_gkg(cfg, logger, force=force)
    meta["finished_at"] = utc_now()
    write_json(COLLECTION_META, meta)
    return meta
