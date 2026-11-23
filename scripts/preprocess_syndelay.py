from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
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

RAW = RAW_DIR / "syndelay" / "repo" / "data" / "syndelay_v1.csv"
OUT_DELAYS = PROCESSED_DIR / "logistics" / "shipment_delays_syndelay.csv"
OUT_FEATURES = PROCESSED_DIR / "logistics" / "logistics_features_syndelay.csv"
OUT_LABELS = TRAINING_DIR / "delay_labels_syndelay.csv"

BASE = datetime(1899, 12, 30)


def excel_serial_to_iso(value: Any) -> str | None:
    try:
        v = float(value)
    except Exception:
        return None
    if pd.isna(v):
        return None
    dt = BASE + timedelta(days=v)
    return dt.replace(microsecond=0).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess SynDelay shipment delay data")
    parser.add_argument("--input", default=str(RAW))
    args = parser.parse_args()
    ensure_dirs()
    logger = setup_logging("preprocess_syndelay")
    df = pd.read_csv(Path(args.input))
    logger.info("Loaded %d rows from %s", len(df), args.input)

    df = df.reset_index(drop=True)
    df["shipment_id"] = (
        "syndelay:"
        + df["order_id"].map(lambda x: str(int(x)) if pd.notna(x) else "na")
        + ":"
        + df["order_item_id"].map(lambda x: str(int(x)) if pd.notna(x) else "na")
        + ":"
        + df.index.map(str)
    )
    df["timestamp"] = df["order_date"].map(excel_serial_to_iso)
    df["planned_shipping_timestamp"] = df["shipping_date"].map(excel_serial_to_iso)

    delays = []
    for _, r in df.iterrows():
        delays.append(
            {
                "shipment_id": r["shipment_id"],
                "asset_id": f"product:{int(r['product_card_id'])}" if pd.notna(r.get("product_card_id")) else None,
                "timestamp": r["timestamp"],
                "latitude": r.get("latitude"),
                "longitude": r.get("longitude"),
                "shipment_status": clean_text(r.get("order_status")),
                "traffic_status": None,
                "waiting_time": None,
                "logistics_delay": int(r["label"]),
                "logistics_delay_reason": None,
                "source_dataset": "syndelay",
                "is_synthetic": 1,
                "delay_class": int(r["label"]),
                "label_semantics": "unverified_integer_class_0_1_2",
                "order_id": str(int(r["order_id"])) if pd.notna(r.get("order_id")) else None,
                "order_item_id": str(int(r["order_item_id"])) if pd.notna(r.get("order_item_id")) else None,
                "shipping_mode": clean_text(r.get("shipping_mode")),
                "market": clean_text(r.get("market")),
                "order_region": clean_text(r.get("order_region")),
                "customer_country": clean_text(r.get("customer_country")),
                "order_country": clean_text(r.get("order_country")),
                "category_name": clean_text(r.get("category_name")),
                "department_name": clean_text(r.get("department_name")),
                "product_name": clean_text(r.get("product_name")),
                "quantity": r.get("order_item_quantity"),
                "sales": r.get("sales"),
                "discount": r.get("order_item_discount"),
                "discount_rate": r.get("order_item_discount_rate"),
                "profit_ratio": r.get("order_item_profit_ratio"),
                "planned_shipping_timestamp": r["planned_shipping_timestamp"],
            }
        )
    delay_fields = [
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
        "delay_class",
        "label_semantics",
        "order_id",
        "order_item_id",
        "shipping_mode",
        "market",
        "order_region",
        "customer_country",
        "order_country",
        "category_name",
        "department_name",
        "product_name",
        "quantity",
        "sales",
        "discount",
        "discount_rate",
        "profit_ratio",
        "planned_shipping_timestamp",
    ]
    n_delay = write_csv(OUT_DELAYS, delays, delay_fields)

    numeric_cols = [
        "profit_per_order",
        "sales_per_customer",
        "order_item_discount",
        "order_item_discount_rate",
        "order_item_product_price",
        "order_item_profit_ratio",
        "order_item_quantity",
        "sales",
        "order_item_total_amount",
        "order_profit_per_order",
        "latitude",
        "longitude",
        "product_price",
    ]
    categorical_cols = [
        "payment_type",
        "category_name",
        "customer_segment",
        "market",
        "order_region",
        "shipping_mode",
        "department_name",
        "customer_country",
        "order_country",
    ]
    features = pd.DataFrame({"shipment_id": df["shipment_id"]})
    for c in numeric_cols:
        if c in df.columns:
            features[c] = pd.to_numeric(df[c], errors="coerce")
    mappings: dict[str, dict[str, int]] = {}
    for c in categorical_cols:
        if c in df.columns:
            values = sorted({clean_text(x) or "" for x in df[c].tolist()})
            mapping = {v: i for i, v in enumerate(values)}
            mappings[c] = mapping
            features[f"{c}_code"] = df[c].map(lambda x: mapping.get(clean_text(x) or "", -1)).astype(int)
    features["is_synthetic"] = 1
    features["source_dataset"] = "syndelay"
    features["data_origin"] = "synthetic"
    feature_fields = list(features.columns)
    n_features = write_csv(OUT_FEATURES, features.to_dict("records"), feature_fields)

    labels = []
    for _, r in df.iterrows():
        labels.append(
            {
                "shipment_id": r["shipment_id"],
                "task": "delay_classification_3class",
                "target_name": "delay_class",
                "target_value": int(r["label"]),
                "target_type": "classification",
                "is_synthetic": 1,
                "label_semantics": "unverified_integer_class_0_1_2",
                "source_dataset": "syndelay",
            }
        )
    label_fields = ["shipment_id", "task", "target_name", "target_value", "target_type", "is_synthetic", "label_semantics", "source_dataset"]
    n_labels = write_csv(OUT_LABELS, labels, label_fields)

    write_json(TRAINING_DIR / "syndelay_category_mappings.json", mappings)
    stats = {
        "rows": len(df),
        "delays_written": n_delay,
        "features_written": n_features,
        "labels_written": n_labels,
        "label_distribution": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "label_semantics": "Repository does not document class meanings; 0/1/2 kept verbatim and flagged as unverified.",
        "excluded_features": ["label", "order_status", "order_date", "shipping_date", "order_id", "order_item_id"],
        "exclusion_reason": "label is the target; order_status is a direct status proxy; dates are not used for advance prediction; IDs are identifiers.",
        "categorical_encoding": "sorted category value -> integer code; -1 for missing/unknown",
        "synthetic_note": "SynDelay is synthetic. Records must not be presented as real shipment outcomes.",
    }
    write_json(PROCESSED_DIR / "logistics" / "syndelay_preprocessing_stats.json", stats)
    logger.info("SynDelay processed: %d rows", len(df))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
