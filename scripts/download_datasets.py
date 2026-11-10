from __future__ import annotations

import argparse
import os
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

from common import (
    DATASETS_DIR,
    append_download_log,
    ensure_dirs,
    human_bytes,
    load_config,
    read_json,
    rel,
    setup_logging,
    sha256_file,
    utc_now,
    write_json,
)
from gdelt_collect import collect as collect_gdelt


def _dest_path(cfg: dict[str, Any], value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else DATASETS_DIR / p


def _sidecar(dest: Path) -> Path:
    return dest.with_name(dest.name + ".download.json")


def _existing_ok(dest: Path, min_bytes: int = 1) -> bool:
    return dest.exists() and dest.is_file() and dest.stat().st_size >= min_bytes


def http_download(session: requests.Session, url: str, dest: Path, cfg: dict[str, Any], logger, force: bool = False) -> dict[str, Any]:
    min_bytes = int(cfg.get("expected_min_bytes", 1))
    if _existing_ok(dest, min_bytes) and not force:
        meta = read_json(_sidecar(dest), default={}) or {}
        return {"status": "skipped_exists", "path": rel(dest), "bytes": dest.stat().st_size, "sha256": meta.get("sha256") or sha256_file(dest)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and force:
        backup = dest.with_name(dest.name + f".bak.{int(time.time())}")
        dest.replace(backup)
        logger.warning("Existing file preserved as %s", backup)
    part = dest.with_name(dest.name + ".part")
    http_cfg = cfg.get("http", {})
    timeout = (int(http_cfg.get("connect_timeout", 20)), int(http_cfg.get("read_timeout", 180)))
    retries = int(http_cfg.get("retries", 4))
    backoff = float(http_cfg.get("backoff_seconds", 5))
    last_error = None
    for attempt in range(retries):
        try:
            headers = {}
            mode = "wb"
            downloaded = 0
            if part.exists() and part.stat().st_size > 0:
                downloaded = part.stat().st_size
                headers["Range"] = f"bytes={downloaded}-"
                mode = "ab"
            with session.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as r:
                if downloaded and r.status_code == 200:
                    mode = "wb"
                    downloaded = 0
                elif downloaded and r.status_code == 416:
                    part.replace(dest)
                    break
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0) or 0) + downloaded
                logger.info("Downloading %s (%s)", url, human_bytes(total) if total else "unknown size")
                last_report = downloaded
                with open(part, mode) as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if downloaded - last_report >= 5 * 1024 * 1024:
                            pct = f" ({downloaded / total * 100:.1f}%)" if total else ""
                            print(f"  {human_bytes(downloaded)}{pct}", file=sys.stderr, flush=True)
                            last_report = downloaded
            if total and part.stat().st_size < min_bytes:
                raise RuntimeError(f"file too small: {part.stat().st_size} bytes")
            part.replace(dest)
            digest = sha256_file(dest)
            side = {
                "url": url,
                "downloaded_at": utc_now(),
                "bytes": dest.stat().st_size,
                "sha256": digest,
                "status": "ok",
            }
            write_json(_sidecar(dest), side)
            return {"status": "ok", "path": rel(dest), "bytes": dest.stat().st_size, "sha256": digest, "url": url}
        except Exception as exc:
            last_error = exc
            wait = backoff * (2 ** attempt)
            logger.warning("Download attempt %d/%d failed for %s: %s", attempt + 1, retries, url, exc)
            if attempt + 1 < retries:
                time.sleep(wait)
    if part.exists():
        part.unlink(missing_ok=True)
    raise RuntimeError(f"download failed after {retries} attempts: {last_error}")


def safe_extract(zip_path: Path, out_dir: Path, logger) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (out_dir / member.filename).resolve()
            if not str(target).startswith(str(out_dir.resolve()) + os.sep):
                raise RuntimeError(f"unsafe zip member: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(rel(target))
    logger.info("Extracted %d file(s) to %s", len(extracted), rel(out_dir))
    return extracted


def mendeley_download(session: requests.Session, name: str, cfg: dict[str, Any], logger, force: bool = False) -> dict[str, Any]:
    dest_dir = _dest_path(cfg, cfg["dest_dir"])
    dest_dir.mkdir(parents=True, exist_ok=True)
    meta_path = dest_dir / "mendeley_metadata.json"
    if meta_path.exists() and not force:
        metadata = read_json(meta_path, default={})
    else:
        r = session.get(cfg["metadata_url"], timeout=60)
        r.raise_for_status()
        metadata = r.json()
        write_json(meta_path, metadata)
    files = metadata.get("files", [])
    expected = files[0] if files else {}
    zip_path = dest_dir / f"{cfg['dataset_id']}-v{cfg['version']}.zip"
    if not zip_path.exists() or force:
        result = http_download(session, cfg["zip_url"], zip_path, {"http": cfg.get("http", {}), "expected_min_bytes": 1}, logger, force=force)
    else:
        result = {"status": "skipped_exists", "path": rel(zip_path), "bytes": zip_path.stat().st_size}
    zip_digest = sha256_file(zip_path)
    extracted = []
    if cfg.get("extract", True):
        extract_dir = dest_dir / "extracted"
        if not extract_dir.exists() or not any(extract_dir.iterdir()) or force:
            extracted = safe_extract(zip_path, extract_dir, logger)
        else:
            extracted = [rel(p) for p in extract_dir.rglob("*") if p.is_file()]
    expected_hash = (expected.get("content_details") or {}).get("sha256_hash")
    payload_hash = None
    if expected.get("filename"):
        candidates = [p for p in dest_dir.rglob(expected["filename"]) if p.is_file()]
        if candidates:
            payload_hash = sha256_file(candidates[0])
    if expected_hash and payload_hash and payload_hash != expected_hash:
        raise RuntimeError(f"Mendeley payload sha256 mismatch: {payload_hash} != {expected_hash}")
    return {
        "status": "ok",
        "path": rel(zip_path),
        "bytes": zip_path.stat().st_size,
        "sha256": zip_digest,
        "payload_sha256": payload_hash,
        "extracted": extracted,
        "metadata": rel(meta_path),
        "dataset_name": metadata.get("name"),
        "doi": (metadata.get("doi") or {}).get("id"),
        "license": (metadata.get("data_licence") or {}).get("short_name"),
    }


def git_clone(cfg: dict[str, Any], logger, force: bool = False) -> dict[str, Any]:
    dest = _dest_path(cfg, cfg["dest_dir"])
    if dest.exists() and any(dest.iterdir()) and not force:
        commit = ""
        try:
            commit = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            pass
        return {"status": "skipped_exists", "path": rel(dest), "commit": commit}
    if dest.exists() and force:
        backup = dest.with_name(dest.name + f".bak.{int(time.time())}")
        dest.replace(backup)
        logger.warning("Existing repository preserved as %s", backup)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", str(int(cfg.get("depth", 1))), cfg["url"], str(dest)]
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    commit = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    pruned = []
    for item in cfg.get("prune_paths", []):
        target = (dest / item).resolve()
        if str(target).startswith(str(dest.resolve()) + os.sep) and target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            pruned.append(item)
            logger.info("Pruned non-essential path from clone: %s", item)
    return {"status": "ok", "path": rel(dest), "commit": commit, "url": cfg["url"], "pruned": pruned}


def local_copy(cfg: dict[str, Any], logger, force: bool = False) -> dict[str, Any]:
    dest = _dest_path(cfg, cfg["dest"])
    if _existing_ok(dest) and not force:
        return {"status": "skipped_exists", "path": rel(dest), "bytes": dest.stat().st_size, "sha256": sha256_file(dest)}
    source = None
    for candidate in cfg.get("search_paths", []):
        p = Path(candidate).expanduser()
        if p.exists() and p.is_file():
            source = p
            break
    if source is None:
        return {"status": "manual_required", "error": "File not found in configured search paths", "dest": rel(dest)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and force:
        backup = dest.with_name(dest.name + f".bak.{int(time.time())}")
        dest.replace(backup)
        logger.warning("Existing file preserved as %s", backup)
    shutil.copy2(source, dest)
    digest = sha256_file(dest)
    return {"status": "ok", "path": rel(dest), "source": str(source), "bytes": dest.stat().st_size, "sha256": digest}


def run_source(name: str, source: dict[str, Any], cfg: dict[str, Any], session: requests.Session, logger, args) -> dict[str, Any]:
    started = time.time()
    entry: dict[str, Any] = {
        "source": name,
        "type": source.get("type"),
        "started_at": utc_now(),
        "restricted": bool(source.get("restricted")),
        "license": source.get("license"),
    }
    try:
        if source.get("restricted") and source.get("requires_license_approval") and not source.get("license_approved") and not args.include_restricted:
            entry.update({"status": "skipped_license_not_approved", "finished_at": utc_now(), "duration_seconds": round(time.time() - started, 2)})
            logger.warning("Skipping %s: restricted licence not approved (use --include-restricted for the isolated restricted folder)", name)
            return entry
        stype = source.get("type")
        if stype == "http_file":
            result = http_download(session, source["url"], _dest_path(cfg, source["dest"]), {**cfg, **source}, logger, force=args.force)
        elif stype == "mendeley":
            result = mendeley_download(session, name, source, logger, force=args.force)
        elif stype == "git":
            result = git_clone(source, logger, force=args.force)
        elif stype == "local_copy":
            result = local_copy(source, logger, force=args.force)
        elif stype == "gdelt":
            result = collect_gdelt(cfg, logger, force=args.force)
        else:
            raise ValueError(f"unsupported source type: {stype}")
        entry.update(result)
        entry["status"] = result.get("status", "ok")
    except Exception as exc:
        entry.update({"status": "failed", "error": str(exc)})
        logger.error("Source %s failed: %s", name, exc)
    entry["finished_at"] = utc_now()
    entry["duration_seconds"] = round(time.time() - started, 2)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Download permitted AtmoGraph datasets")
    parser.add_argument("--config", default=None)
    parser.add_argument("--only", default=None, help="Comma-separated source names")
    parser.add_argument("--force", action="store_true", help="Re-download and preserve old files as .bak")
    parser.add_argument("--include-restricted", action="store_true", help="Download restricted sources into the isolated restricted folder")
    args = parser.parse_args()
    ensure_dirs()
    logger = setup_logging("download")
    cfg = load_config(Path(args.config) if args.config else None)
    cfg.setdefault("http", {})
    sources = cfg.get("sources", {})
    only = {s.strip() for s in args.only.split(",")} if args.only else None
    session = requests.Session()
    session.headers.update({"User-Agent": cfg["http"].get("user_agent", "AtmoGraph-research/1.0")})
    entries = []
    for name, source in sources.items():
        if only and name not in only:
            continue
        if not source.get("enabled", False) and not (only and name in only):
            logger.info("Skipping disabled source: %s", name)
            entries.append({"source": name, "status": "disabled", "finished_at": utc_now()})
            continue
        logger.info("=== %s ===", name)
        entry = run_source(name, source, cfg, session, logger, args)
        append_download_log(entry)
        entries.append(entry)
        logger.info("%s -> %s", name, entry.get("status"))
    ok = sum(1 for e in entries if e.get("status") in {"ok", "skipped_exists"})
    failed = [e["source"] for e in entries if e.get("status") == "failed"]
    restricted = [e["source"] for e in entries if e.get("restricted")]
    print(json.dumps({"sources": entries, "ok": ok, "failed": failed, "restricted": restricted}, indent=2, default=str))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
