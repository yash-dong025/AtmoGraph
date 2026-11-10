from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    PROCESSED_DIR,
    RAW_DIR,
    TRAINING_DIR,
    clean_text,
    ensure_dirs,
    setup_logging,
    write_csv,
    write_json,
)

RAW = RAW_DIR / "existing_smart_logistics" / "smart_logistics_dataset.csv"
OUT_DELAYS = PROCESSED_DIR / "logistics" / "shipment_delays_smart_logistics.csv"
OUT_FEATURES = PROCESSED_DIR / "logistics" / "logistics_features_smart_logistics.csv"
OUT_LABELS = TRAINING_DIR / "delay_labels_smart_logistics.csv"

COMBINED_DELAYS = PROCESSED_DIR / "logistics" / "shipment_delays.csv"
COMBINED_FEATURES = PROCESSED_DIR / "logistics" / "logistics_features.csv"
COMBINED_LABELS = TRAINING_DIR / "delay_labels.csv"

DELAY_FIELDS = [
    "shipment_id",
    "asset_id",
    "timestamp",
    "latitude",
    "longitude",
    "shipment_status",
    "traffic_status",
    "waiting_time",
    "logistics_delay",
    "logistics_delay_reason",
    "source_dataset",
    "is_synthetic",
    "data_origin",
    "delay_class",
    "label_semantics",
]

FEATURE_FIELDS = [
    "shipment_id",
    "source_dataset",
    "latitude",
    "longitude",
    "waiting_time",
    "inventory_level",
    "temperature",
    "humidity",
    "asset_utilization",
    "demand_forecast",
    "traffic_status_code",
    "asset_id_code",
    "is_synthetic",
    "data_origin",
]

LABEL_FIELDS = [
    "shipment_id",
    "task",
    "target_name",
    "target_value",
    "target_type",
    "is_synthetic",
    "label_semantics",
    "source_dataset",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess the supplied smart logistics dataset")
    parser.add_argument("--input", default=str(RAW))
    args = parser.parse_args()
    ensure_dirs()
    logger = setup_logging("preprocess_smart_logistics")
    path = Path(args.input)
    if not path.exists():
        logger.error("Missing %s; supply the file manually under datasets/raw/existing_smart_logistics/", path)
        print(json.dumps({"status": "manual_required", "expected_path": str(path)}, indent=2))
        return 2
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    df["shipment_id"] = (
        "smartlog:" + df["Asset_ID"].astype(str) + ":" + df["Timestamp"].astype(str).str.replace(r"[^0-9]", "", regex=True)
    )
    traffic_values = sorted({clean_text(x) or "" for x in df["Traffic_Status"].tolist()})
    traffic_map = {v: i for i, v in enumerate(traffic_values)}
    asset_values = sorted({clean_text(x) or "" for x in df["Asset_ID"].tolist()})
    asset_map = {v: i for i, v in enumerate(asset_values)}

    delays = []
    for _, r in df.iterrows():
        delays.append(
            {
                "shipment_id": r["shipment_id"],
                "asset_id": clean_text(r.get("Asset_ID")),
                "timestamp": clean_text(r.get("Timestamp")),
                "latitude": r.get("Latitude"),
                "longitude": r.get("Longitude"),
                "shipment_status": clean_text(r.get("Shipment_Status")),
                "traffic_status": clean_text(r.get("Traffic_Status")),
                "waiting_time": r.get("Waiting_Time"),
                "logistics_delay": int(r["Logistics_Delay"]),
                "logistics_delay_reason": clean_text(r.get("Logistics_Delay_Reason")),
                "source_dataset": "smart_logistics",
                "is_synthetic": None,
                "data_origin": "local_supplied_unverified",
                "delay_class": int(r["Logistics_Delay"]),
                "label_semantics": "binary_Logistics_Delay_0_1",
            }
        )
    n_delays = write_csv(OUT_DELAYS, delays, DELAY_FIELDS)

    features = []
    for _, r in df.iterrows():
        features.append(
            {
                "shipment_id": r["shipment_id"],
                "source_dataset": "smart_logistics",
                "latitude": r.get("Latitude"),
                "longitude": r.get("Longitude"),
                "waiting_time": r.get("Waiting_Time"),
                "inventory_level": r.get("Inventory_Level"),
                "temperature": r.get("Temperature"),
                "humidity": r.get("Humidity"),
                "asset_utilization": r.get("Asset_Utilization"),
                "demand_forecast": r.get("Demand_Forecast"),
                "traffic_status_code": traffic_map.get(clean_text(r.get("Traffic_Status")) or "", -1),
                "asset_id_code": asset_map.get(clean_text(r.get("Asset_ID")) or "", -1),
                "is_synthetic": None,
                "data_origin": "local_supplied_unverified",
            }
        )
    n_features = write_csv(OUT_FEATURES, features, FEATURE_FIELDS)

    labels = []
    for _, r in df.iterrows():
        labels.append(
            {
                "shipment_id": r["shipment_id"],
                "task": "delay_classification_binary",
                "target_name": "logistics_delay",
                "target_value": int(r["Logistics_Delay"]),
                "target_type": "classification",
                "is_synthetic": None,
                "label_semantics": "binary_Logistics_Delay_0_1",
                "source_dataset": "smart_logistics",
            }
        )
    n_labels = write_csv(OUT_LABELS, labels, LABEL_FIELDS)

    write_json(
        PROCESSED_DIR / "logistics" / "smart_logistics_category_mappings.json",
        {"traffic_status": traffic_map, "asset_id": asset_map},
    )

    combine(logger)
    stats = {
        "rows": len(df),
        "delays_written": n_delays,
        "features_written": n_features,
        "labels_written": n_labels,
        "label_distribution": {str(k): int(v) for k, v in df["Logistics_Delay"].value_counts().sort_index().items()},
        "excluded_features": ["Shipment_Status", "Logistics_Delay_Reason", "Logistics_Delay", "Timestamp"],
        "exclusion_reason": "Shipment_Status='Delayed' directly encodes the target; reason is post-hoc; timestamp is not used for advance prediction.",
        "data_origin": "Supplied locally; provenance not documented. Not assumed to be real or synthetic.",
    }
    write_json(PROCESSED_DIR / "logistics" / "smart_logistics_preprocessing_stats.json", stats)
    logger.info("Smart logistics processed: %d rows", len(df))
    print(json.dumps(stats, indent=2))
    return 0


def _read(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def combine(logger) -> None:
    syndelay_delays = _read(PROCESSED_DIR / "logistics" / "shipment_delays_syndelay.csv")
    if not syndelay_delays:
        syndelay_delays = _read(PROCESSED_DIR / "logistics" / "shipment_delays.csv")
        syndelay_delays = [r for r in syndelay_delays if r.get("source_dataset") == "syndelay"]
    smart_delays = _read(OUT_DELAYS)
    combined = []
    for r in syndelay_delays + smart_delays:
        combined.append(
            {
                "shipment_id": r.get("shipment_id"),
                "asset_id": r.get("asset_id"),
                "timestamp": r.get("timestamp"),
                "latitude": r.get("latitude"),
                "longitude": r.get("longitude"),
                "shipment_status": r.get("shipment_status"),
                "traffic_status": r.get("traffic_status"),
                "waiting_time": r.get("waiting_time"),
                "logistics_delay": r.get("logistics_delay"),
                "logistics_delay_reason": r.get("logistics_delay_reason"),
                "source_dataset": r.get("source_dataset"),
                "is_synthetic": r.get("is_synthetic"),
                "data_origin": r.get("data_origin") or ("synthetic" if r.get("source_dataset") == "syndelay" else "local_supplied_unverified"),
                "delay_class": r.get("delay_class"),
                "label_semantics": r.get("label_semantics"),
            }
        )
    write_csv(COMBINED_DELAYS, combined, DELAY_FIELDS)

    syndelay_features = _read(PROCESSED_DIR / "logistics" / "logistics_features_syndelay.csv")
    if not syndelay_features:
        syndelay_features = _read(PROCESSED_DIR / "logistics" / "logistics_features.csv")
        syndelay_features = [r for r in syndelay_features if r.get("source_dataset") == "syndelay"]
    smart_features = _read(OUT_FEATURES)
    frows = []
    for r in syndelay_features + smart_features:
        frows.append(
            {
                "shipment_id": r.get("shipment_id"),
                "source_dataset": r.get("source_dataset"),
                "latitude": r.get("latitude"),
                "longitude": r.get("longitude"),
                "waiting_time": r.get("waiting_time"),
                "inventory_level": r.get("inventory_level"),
                "temperature": r.get("temperature"),
                "humidity": r.get("humidity"),
                "asset_utilization": r.get("asset_utilization"),
                "demand_forecast": r.get("demand_forecast"),
                "traffic_status_code": r.get("traffic_status_code"),
                "asset_id_code": r.get("asset_id_code"),
                "is_synthetic": r.get("is_synthetic"),
                "data_origin": r.get("data_origin") or ("synthetic" if r.get("source_dataset") == "syndelay" else "local_supplied_unverified"),
            }
        )
    write_csv(COMBINED_FEATURES, frows, FEATURE_FIELDS)

    syndelay_labels = _read(TRAINING_DIR / "delay_labels_syndelay.csv")
    smart_labels = _read(OUT_LABELS)
    lrows = []
    for r in syndelay_labels + smart_labels:
        lrows.append({k: r.get(k) for k in LABEL_FIELDS})
    write_csv(COMBINED_LABELS, lrows, LABEL_FIELDS)
    logger.info("Combined logistics: %d shipments, %d features, %d labels", len(combined), len(frows), len(lrows))


if __name__ == "__main__":
    raise SystemExit(main())
