# AtmoGraph Dataset Package

Reproducible dataset acquisition, validation, and preprocessing for **AtmoGraph: Supply Chain Ripple Effect Predictor**.

This package prepares:

- a combined company supply-chain graph (SAMPOQ + IC-SPLC-v2 + AnonNet),
- node features and IC-SPLC embeddings,
- reproducible node/edge/delay splits,
- synthetic and locally supplied shipment-delay tasks,
- GDELT news metadata for disruption monitoring,
- restricted evaluation scenarios, ground truth, and benchmark assets.

It does **not** contain verified company-level risk labels or confirmed disruption outcomes. Do not claim real-world delay-prediction capability from these files alone.

## Quick start

```bash
python3 -m pip install -r requirements-data.txt

python3 scripts/download_datasets.py
python3 scripts/preprocess_sampoq.py
python3 scripts/preprocess_ic_splc.py
python3 scripts/preprocess_anonnet.py
python3 scripts/preprocess_syndelay.py
python3 scripts/preprocess_smart_logistics.py
python3 scripts/preprocess_gdelt.py
python3 scripts/build_graph_dataset.py
python3 scripts/inspect_datasets.py
python3 scripts/validate_datasets.py
python3 scripts/generate_dataset_report.py
```

Restricted CC BY-NC material is processed only with explicit approval:

```bash
python3 scripts/preprocess_disruption_monitoring.py --allow-restricted
```

Or run the orchestrator:

```bash
python3 scripts/run_pipeline.py --skip-download
python3 scripts/run_pipeline.py --allow-restricted
```

IC-SPLC extraction from the Neo4j dump requires a running Neo4j 5.x instance:

```bash
python3 scripts/preprocess_ic_splc.py --extract --uri neo4j://127.0.0.1:7687
```

The extracted CSVs are already included in this package, so normal preprocessing does not need Neo4j.

## Directory layout

```
datasets/
  raw/                     acquired source data (licences apply)
  processed/
    companies/             per-source and combined company tables
    relationships/         per-source and combined edge tables (+ Parquet)
    graph/                 graph nodes, edges, node features, metadata
    logistics/             shipment records and model-ready features
    disruptions/           GDELT news metadata and event mentions
    restricted/            CC BY-NC material, excluded from main pipeline
  training/                indices, features, labels, embeddings, splits
  evaluation/              restricted scenarios, ground truth, benchmark
  metadata/                manifest, download log, data dictionary, licences
  reports/                 inventory, validation, summary, quality reports
config/dataset_config.yaml source definitions and parameters
scripts/                   acquisition, preprocessing, validation, reporting
```

## Dataset inventory

| Dataset | Rows | Licence | Use |
|---|---|---|---|
| SAMPOQ | 13,372 companies / 16,102 edges | CC BY 4.0 | Main graph |
| IC-SPLC-v2 | 109,403 nodes / 115,767 edge rows | CC BY 4.0 | Main graph + 128-d embeddings |
| AnonNet | 7,245 nodes / 43,922 edges | No LICENSE file | Main graph |
| SynDelay | 155,488 synthetic shipments | No LICENSE file | 3-class delay task |
| Smart Logistics | 1,000 shipments | Provenance unknown | Binary delay task |
| GDELT | 847 articles / 866 mentions | GDELT terms | Unverified disruption signal |
| Disruption Monitoring | 6,222 companies / 16,640 edges | CC BY-NC 4.0 | Restricted evaluation only |

Combined graph: **130,020 nodes**, **166,690 unique directed edges**, **23,198 weakly connected components** (largest 52,473 nodes), **16 isolated nodes**.

## Key outputs

- `datasets/processed/companies/companies.csv`
- `datasets/processed/relationships/supply_chain_edges.csv` / `.parquet`
- `datasets/processed/graph/{nodes.csv, edges.csv, node_features.csv, graph_metadata.json}`
- `datasets/processed/logistics/{shipment_delays.csv, logistics_features.csv}`
- `datasets/processed/disruptions/{news_articles.csv, disruption_events.csv}`
- `datasets/training/{node_features.csv, node_index_mapping.csv, edge_index.csv, risk_labels.csv, delay_labels.csv, node_embeddings.parquet, dataset_splits.json}`
- `datasets/evaluation/` (restricted)
- `datasets/metadata/dataset_manifest.json`
- `datasets/reports/{dataset_summary.md, data_quality_report.md, preprocessing_report.md}`

Full column-level documentation: `datasets/metadata/data_dictionary.md`.

## Using the graph

Node IDs are namespaced (`sampoq:`, `icsplc:`, `anonnet:`) and must not be merged across anonymised sources.

Load into Neo4j:

```cypher
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
CREATE (:Company {
  id: row.company_id,
  name: row.company_name,
  country: row.country,
  industry: row.industry,
  source: row.source_dataset
});

LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
MATCH (s:Company {id: row.source_company_id})
MATCH (t:Company {id: row.target_company_id})
CREATE (s)-[:SUPPLIES_TO {
  edge_id: row.edge_id,
  relationship_type: row.relationship_type,
  confidence: toFloat(row.confidence)
}]->(t);
```

PyTorch Geometric:

```python
import pandas as pd
import torch
from torch_geometric.data import Data

edges = pd.read_csv("datasets/training/edge_index.csv")
features = pd.read_csv("datasets/processed/graph/node_features.csv")
edge_index = torch.tensor(edges[["source_index", "target_index"]].values.T, dtype=torch.long)
x = torch.tensor(features.drop(columns=["node_id", "node_type"]).values, dtype=torch.float)
data = Data(x=x, edge_index=edge_index)
```

For embeddings, load `datasets/training/node_embeddings.parquet` and respect `embedding_available`; missing rows are zero-filled.

## Validation and reports

- `python3 scripts/validate_datasets.py` runs 40 integrity checks (endpoints, self-loops, split coverage/disjointness, label availability, leakage, restricted exclusion).
- `datasets/reports/validation_results.json` records every check.
- `datasets/reports/data_quality_report.md` documents missingness, fragmentation, duplicate handling, and source-specific issues.
- `datasets/metadata/dataset_manifest.json` records SHA-256 hashes for processed, training, evaluation, metadata, reports, and key raw assets.

## Data quality and label caveats

- **No company risk labels.** `training/risk_labels.csv` is entirely `not_available`. Do not train or report a company risk model on these data.
- **SynDelay labels are undocumented.** Classes 0/1/2 are kept verbatim and flagged. Do not interpret them as low/medium/high severity.
- **Smart Logistics provenance is unknown.** Do not assume the records are real or synthetic.
- **IC-SPLC and AnonNet are anonymised.** No cross-source entity resolution was performed.
- **GDELT records are unverified news mentions.** They are not confirmed disruptions and have no company attribution.
- **Graph is fragmented.** 23,198 components; the largest covers 40.4% of nodes. Many components are size 2.
- **Embedding coverage is 24%.** Only 31,182 of 130,020 nodes have IC-SPLC embeddings.
- **SAMPOQ lacks country/industry/product fields.** Those columns are empty for its entities.

## Licence restrictions

| Dataset | Licence | Action |
|---|---|---|
| SAMPOQ | CC BY 4.0 | Attribution required. |
| IC-SPLC-v2 | CC BY 4.0 | Attribution required. |
| AnonNet | No LICENSE file | Obtain permission before redistribution/commercial use. |
| SynDelay | No LICENSE file | Obtain permission before redistribution/commercial use. |
| Smart Logistics | Not specified | Confirm provenance and licence. |
| GDELT | GDELT 2.0 terms | Review terms; metadata only. |
| Disruption Monitoring | CC BY-NC 4.0 | Non-commercial only; keep restricted. |

Restricted material is stored under `raw/disruption_monitoring/restricted/`, `processed/restricted/`, and `evaluation/**/restricted/`. It is excluded from the main graph and training data.

See `datasets/metadata/source_licenses.md` for details. This is not legal advice.

## Attribution

- SAMPOQ Supply Chain 10-K dataset, https://huggingface.co/datasets/SAMPOQ/supply-chain-10k (CC BY 4.0).
- King, P. (2026). IC-SPLC-v2: A Supply Chain Structure Neo4j Data. Mendeley Data, V2. doi:10.17632/txh3rrsh89.2 (CC BY 4.0).
- Xu, L., Long, Y., Brintrup, A. (2025). SynDelay and AnonNet (licence unspecified).
- The GDELT Project, https://www.gdeltproject.org/.
- AlMahri, S. (2025-2026). supply-chain-disruption-monitoring (CC BY-NC 4.0).
- Locally supplied `smart_logistics_dataset.csv` (provenance to be confirmed).
