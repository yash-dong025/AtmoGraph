import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ml.preprocessing.feature_engineering import build_edge_features, build_node_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_node_fixture(tmp_path: Path) -> dict[str, Path]:
    node_ids = ["n1", "n2", "n3", "n4"]
    node_index = pd.DataFrame({"node_id": node_ids, "node_index": range(4)})
    node_index_path = tmp_path / "node_index.csv"
    node_index.to_csv(node_index_path, index=False)

    rows = []
    for index, node_id in enumerate(node_ids):
        rows.append(
            {
                "node_id": node_id,
                "node_type": "company",
                "component_id": "c1",
                "country_code": [1, -1, 2, -1][index],
                "industry_code": [0, 1, -1, 2][index],
                "product_category_code": [3, -1, 4, 5][index],
                "is_isolated": 0,
                "source_sampoq": int(index == 0),
                "source_ic_splc": int(index != 0),
                "source_anonnet": 0,
                "has_country": int(index != 1),
                "has_industry": 1,
                "has_product_category": 1,
                "embedding_available": int(index in {0, 2}),
                "is_anonymised": int(index != 0),
                "degree": [0, 3, 1000, 1][index],
                "in_degree": 0,
                "out_degree": 0,
                "raw_degree": [0, 3, 1000, 1][index],
                "component_size": 4,
                "first_edge_year": 2020,
                "last_edge_year": 2021,
            }
        )
    node_features_path = tmp_path / "node_features.csv"
    pd.DataFrame(rows).to_csv(node_features_path, index=False)

    embedding_values = {
        f"emb_{dim}": [
            (np.linspace(0.0, 1.0, 4) + dim)[index]
            if index in {0, 2}
            else np.nan
            for index in range(4)
        ]
        for dim in range(128)
    }
    embeddings = pd.concat(
        [
            pd.DataFrame(
                {
                    "node_id": node_ids,
                    "embedding_available": [True, False, True, False],
                }
            ),
            pd.DataFrame(embedding_values),
        ],
        axis=1,
    )
    embeddings_path = tmp_path / "embeddings.parquet"
    embeddings.to_parquet(embeddings_path, index=False)

    splits_path = tmp_path / "splits.json"
    splits_path.write_text(
        json.dumps(
            {
                "node_splits": {
                    "train": ["n1", "n2"],
                    "val": ["n3"],
                    "test": ["n4"],
                }
            }
        )
    )
    return {
        "node_features": node_features_path,
        "node_index": node_index_path,
        "embeddings": embeddings_path,
        "splits": splits_path,
    }


def test_node_features_have_stable_contract_and_train_only_fill(tmp_path):
    paths = _write_node_fixture(tmp_path)
    bundle = build_node_features(
        node_features_path=paths["node_features"],
        node_index_path=paths["node_index"],
        embeddings_path=paths["embeddings"],
        splits_path=paths["splits"],
    )
    assert bundle.x.shape == (4, 146)
    assert bundle.x_num.shape == (4, 18)
    assert bundle.x_emb.shape == (4, 128)
    assert bundle.x_cat.shape == (4, 3)
    assert bundle.feature_names[-1] == "icsplc_emb_127"
    assert bundle.emb_mask.tolist() == [True, False, True, False]
    assert np.allclose(bundle.x_emb[[1, 3]], 0.0)
    assert bundle.x_cat[:, 0].tolist() == [2, 0, 3, 0]
    assert bundle.train_mask.tolist() == [True, True, False, False]
    assert bundle.val_mask.tolist() == [False, False, True, False]
    assert bundle.test_mask.tolist() == [False, False, False, True]
    assert bundle.medians["log1p_degree"] == math.log(2.0)
    assert np.isfinite(bundle.x).all()


def test_edge_features_have_stable_contract(tmp_path):
    edges = pd.DataFrame(
        [
            {
                "edge_id": "e1",
                "source_company_id": "n1",
                "first_year": 2020,
                "last_year": 2021,
                "edge_count": 2,
                "confidence": 0.9,
                "relationship_type": "supplies_product_to",
                "source_datasets": "ic_splc;sampoq",
            },
            {
                "edge_id": "e2",
                "source_company_id": "n2",
                "first_year": None,
                "last_year": None,
                "edge_count": 1,
                "confidence": 0.5,
                "relationship_type": "supplier_to_customer",
                "source_datasets": "ic_splc",
            },
            {
                "edge_id": "e3",
                "source_company_id": "n3",
                "first_year": 2019,
                "last_year": 2021,
                "edge_count": 4,
                "confidence": 0.7,
                "relationship_type": "supply_relationship_anonymised",
                "source_datasets": "anonnet",
            },
        ]
    )
    edge_index = pd.DataFrame(
        [
            {"edge_id": "e1", "source_index": 0, "target_index": 1, "split": "train"},
            {"edge_id": "e2", "source_index": 1, "target_index": 2, "split": "val"},
            {"edge_id": "e3", "source_index": 2, "target_index": 3, "split": "test"},
        ]
    )
    edges_path = tmp_path / "edges.csv"
    edge_index_path = tmp_path / "edge_index.csv"
    edges.to_csv(edges_path, index=False)
    edge_index.to_csv(edge_index_path, index=False)

    bundle = build_edge_features(edges_path=edges_path, edge_index_path=edge_index_path)
    assert bundle.edge_attr.shape == (3, 12)
    assert bundle.edge_index.tolist() == [[0, 1, 2], [1, 2, 3]]
    assert bundle.train_mask.tolist() == [True, False, False]
    assert bundle.val_mask.tolist() == [False, True, False]
    assert bundle.test_mask.tolist() == [False, False, True]
    assert np.isfinite(bundle.edge_attr).all()


def test_original_splits_are_disjoint_and_match_metadata():
    splits = json.loads((PROJECT_ROOT / "datasets/training/dataset_splits.json").read_text())
    metadata = json.loads(
        (PROJECT_ROOT / "ml/experiments/graphsage_link_prediction/dataset_metadata.json").read_text()
    )
    for key, section in (("node", "node_splits"), ("edge", "edge_splits")):
        split_sets = {name: set(ids) for name, ids in splits[section].items()}
        assert not split_sets["train"] & split_sets["val"]
        assert not split_sets["train"] & split_sets["test"]
        assert not split_sets["val"] & split_sets["test"]
        assert {name: len(ids) for name, ids in split_sets.items()} == metadata[f"{key}_split_sizes"]
