# AtmoGraph Dataset Summary

Generated: 2026-11-10T05:42:28Z

## Scope

This dataset package supports a supply-chain ripple-effect research prototype. It combines real company relationship sources, anonymised networks, synthetic shipment records, unverified news metadata, and a restricted evaluation benchmark. It does **not** contain verified company-level risk labels or confirmed disruption outcomes.

## Source overview

| Dataset | Content | Size | Licence | Use |
|---|---|---|---|---|
| SAMPOQ | Supplier/customer relations from US 10-K filings | 13,372 companies / 16,102 edges | CC BY 4.0 | Main graph |
| IC-SPLC-v2 | Neo4j supply-chain structure, anonymised | 109,403 nodes / 115,767 edges | CC BY 4.0 | Main graph + 128-d embeddings |
| AnonNet | Anonymised real-world supply network | 7,245 nodes / 43,922 edges | No LICENSE file | Main graph |
| SynDelay | Synthetic shipment delivery-delay records | 155,488 rows | No LICENSE file | Delay task (synthetic) |
| Smart Logistics | Locally supplied shipment records | 1,000 rows | Provenance unknown | Delay task (binary) |
| GDELT | News metadata via DOC API + GKG fallback | 847 articles / 866 mentions | GDELT terms | Disruption monitoring (unverified) |
| Disruption Monitoring | Curated KG + scenarios + benchmark | 6,222 companies / 16,640 edges | CC BY-NC 4.0 | Restricted evaluation only |

## Graph snapshot

- Companies (nodes): **130,020**
- Raw relationship rows: **175,754**
- Unique directed graph edges after duplicate collapse: **166,690**
- Duplicate edge rows collapsed: **9,064**
- Duplicate edge IDs merged during combination: **37**
- Weakly connected components: **23,198**
- Largest component: **52,473** (40.4%)
- Isolated nodes: **16**
- Nodes with IC-SPLC embeddings: **31,182** (24.0%)
- Degree: median **1.0**, p99 **28.0**, max **1402**

## Disruption monitoring

- News articles: **847** from **493** domains.
- Event mentions: **866**.
- Date range: **2026-11-03** to **2026-11-10**.
- Collection: DOC API **340** articles; GKG fallback **507** articles.
- All records are **unverified news mentions**, not confirmed disruptions.

## Shipment delay tasks

| Task | Class | Rows |
|---|---|---|
| delay_classification_3class | 0 | 36,724 |
| delay_classification_3class | 1 | 29,055 |
| delay_classification_3class | 2 | 89,709 |
| delay_classification_binary | 0 | 434 |
| delay_classification_binary | 1 | 566 |

SynDelay classes 0/1/2 are **undocumented** and are not interpreted as delay severity. Smart Logistics binary labels are direct but the file provenance is unknown.

## Evaluation assets (restricted)

- Scenarios: **30** synthetic news scenarios with ground-truth ripple targets.
- Benchmark items: **1,117** relevance items.
- Licence: **CC BY-NC 4.0**; non-commercial use only. Excluded from the main graph and training data.

## Real, synthetic, inferred

| Source | Status | Confidence | Notes |
|---|---|---|---|
| SAMPOQ | Real, self-reported | 0.9 | Extracted from SEC 10-K filings; not independently audited. |
| IC-SPLC | Real-world structure, anonymised + LLM-extracted | 0.7 | Entity IDs anonymised; direction per dataset card. |
| AnonNet | Real-world structure, anonymised | 0.8 | Direction not formally documented. |
| SynDelay | Synthetic | n/a | Do not present as real shipment outcomes. |
| Smart Logistics | Provenance unknown | n/a | Do not assume real or synthetic. |
| GDELT | Real news metadata | n/a | Mentions only; no confirmed event attribution. |
| Disruption Monitoring | Curated KG; scenarios synthetic | n/a | CC BY-NC restricted. |

## Label availability

- Company-level risk labels: **not available**. `training/risk_labels.csv` is intentionally empty and marked `not_available`.
- Shipment delay labels: available for 156,488 rows (synthetic 3-class + binary).
- Disruption relevance labels: available only in the restricted benchmark.

## Key caveats

- No cross-source entity resolution: IC-SPLC and AnonNet IDs are anonymised and must not be merged with identifiable companies.
- The graph is structurally fragmented: many small components; the largest component covers less than half of all nodes.
- GDELT records are unverified and the collection window is only eight days.
- Restricted CC BY-NC data must not enter commercial deliverables.
