from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"
TRAINING_DIR = DATASETS_DIR / "training"
EVALUATION_DIR = DATASETS_DIR / "evaluation"
METADATA_DIR = DATASETS_DIR / "metadata"
REPORTS_DIR = DATASETS_DIR / "reports"
CONFIG_PATH = PROJECT_ROOT / "config" / "dataset_config.yaml"
LOG_PATH = METADATA_DIR / "pipeline.log"

UTC = timezone.utc


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dirs() -> None:
    for p in [
        RAW_DIR,
        PROCESSED_DIR / "companies",
        PROCESSED_DIR / "relationships",
        PROCESSED_DIR / "disruptions",
        PROCESSED_DIR / "logistics",
        PROCESSED_DIR / "graph",
        TRAINING_DIR,
        EVALUATION_DIR / "scenarios",
        EVALUATION_DIR / "ground_truth",
        METADATA_DIR,
        REPORTS_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str = "atmograph") -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")
    tmp.replace(path)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human_bytes(n: int | float | None) -> str:
    if n is None:
        return "unknown"
    n = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def normalise_name(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def slugify(value: Any, max_len: int = 60) -> str:
    s = normalise_name(value).replace(" ", "-")
    return s[:max_len].strip("-") or "unknown"


def stable_id(prefix: str, key: Any, length: int = 16) -> str:
    raw = f"{prefix}|{key}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:length]
    return f"{prefix}_{digest}"


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).replace("\u00a0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s or None


def parse_date(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, (int, float)):
        try:
            v = float(value)
        except Exception:
            return None
        if 20000 <= v <= 80000:
            base = datetime(1899, 12, 30, tzinfo=UTC)
            from datetime import timedelta

            return (base + timedelta(days=v)).date().isoformat()
        return None
    s = str(value).strip()
    if not s:
        return None
    if re.fullmatch(r"\d{8}", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ].*", s):
        return s[:10]
    for fmt in ["%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y%m%d"]:
        try:
            return datetime.strptime(s[:10], fmt).date().isoformat()
        except Exception:
            continue
    return None


def read_csv_rows(path: Path, delimiter: str = ",", encoding: str = "utf-8") -> list[dict[str, str]]:
    with open(path, "r", encoding=encoding, errors="replace", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})
            n += 1
    return n


def append_download_log(entry: dict[str, Any]) -> None:
    path = METADATA_DIR / "download_log.json"
    data = read_json(path, default={"entries": []})
    if isinstance(data, list):
        data = {"entries": data}
    data.setdefault("entries", []).append(entry)
    data["updated_at"] = utc_now()
    write_json(path, data)


def dataset_manifest() -> dict[str, Any]:
    path = METADATA_DIR / "dataset_manifest.json"
    return read_json(path, default={"generated_at": None, "sources": {}, "files": []})


def save_dataset_manifest(data: dict[str, Any]) -> None:
    data["generated_at"] = utc_now()
    write_json(METADATA_DIR / "dataset_manifest.json", data)


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file())


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}
