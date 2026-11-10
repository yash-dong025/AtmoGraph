# Data Quality Report

Generated: 2026-11-10T05:42:28Z

## Validation

- Status: **pass**
- Checks passed: **40**
- Checks failed: **0**

See `reports/validation_results.json` for the full check list.

## Graph structure

| Metric | Value |
|---|---|
| Nodes | 130,020 |
| Raw edge rows | 175,754 |
| Unique directed graph edges | 166,690 |
| Duplicate edge rows collapsed | 9,064 |
| Duplicate edge IDs merged | 37 |
| Self-loops remaining | 0 |
| Missing endpoint edges | 0 |
| Weakly connected components | 23,198 |
| Largest component | 52,473 (40.4%) |
| Isolated nodes | 16 |
| Degree median / p99 / max | 1.0 / 28.0 / 1402 |

Component-size distribution (component size: count): 1:16, 2:16909, 3:3337, 4:1390, 5:632, 6:319, 7:167, 8:99, 9:83, 10:51, 11:37, 12:33. The ten largest components contain 52473, 9457, 7245, 139, 56, 51, 39, 37, 37, 33 nodes respectively. Fragmentation is expected because the three sources are not cross-resolved and include many small supplier/customer clusters.

## Missingness in the combined tables

| Table | Field | Missing rows | Share |
|---|---|---|---|
| companies.csv | country | 99,143 | 76.3% |
| companies.csv | industry | 119,455 | 91.9% |
| companies.csv | product_category | 121,467 | 93.4% |
| companies.csv | ticker | 126,647 | 97.4% |
| supply_chain_edges.csv | year | 43,922 | 25.0% |
| supply_chain_edges.csv | filing_date | 159,652 | 90.8% |
| supply_chain_edges.csv | product_category | 114,218 | 65.0% |

SAMPOQ has no country, industry, or product fields in the source, so those values are empty for its 13,372 companies. IC-SPLC and AnonNet contribute most of the missing year/filing-date values.

## Source-specific issues

### SAMPOQ
- Real, self-reported 10-K relationships. Direction was derived from `relation_type` and standardised to Supplier -> Customer.
- 4 duplicate endpoint/date pairs were observed and flagged rather than silently removed.
- Company identifiers use ticker, CIK, or hashed normalised names; name-only entities may be ambiguous.
- No country/industry/product metadata exists in this source.

### IC-SPLC-v2
- 109,403 anonymised nodes and 115,767 edge rows, of which **37 duplicate edge IDs** exist in the processed source file. The combined table merges them by `edge_id`.
- Only **31,182** nodes have 128-d embeddings (28.5% of IC-SPLC nodes; 24.0% of the full graph).
- Product category is missing for 96,831 of 115,767 edge rows; filing date is entirely absent.
- Relationships are LLM-extracted from an anonymised corpus; direction confidence is 0.7.

### AnonNet
- 7,245 nodes and 43,926 edges; **4 self-loops** were removed.
- Country is missing for 1,490 nodes (20.6%); products are missing for 537 nodes.
- Edge direction is preserved as provided; the source does not publish a formal direction definition.
- Anonymised IDs must not be merged with identifiable company records.

### SynDelay
- 155,488 synthetic rows. Class distribution: 0 = 36,724; 1 = 29,055; 2 = 89,709.
- Class meanings are **not documented**. The labels are retained verbatim and flagged `unverified_integer_class_0_1_2`.
- 41 duplicate `order_id`/`order_item_id` pairs were made unique with a row suffix to avoid split leakage.
- No LICENSE file; rights are unspecified.

### Smart Logistics
- 1,000 locally supplied rows; binary labels 0 = 434, 1 = 566.
- Provenance is not documented. Do not assume the records are real or synthetic.
- `Shipment_Status`, `Logistics_Delay_Reason`, `Logistics_Delay`, and `Timestamp` are excluded as leakage or post-hoc fields.

### GDELT
- 847 articles and 866 event mentions from 493 domains over 2026-11-03 to 2026-11-10.
- The DOC API returned HTTP 429 for `factory shutdown`; 7 of 8 keyword queries ultimately succeeded. A GKG sample supplied 507 additional articles.
- GKG theme matches are noisier than phrase matches; only supply-chain-specific themes are retained.
- Source country is missing for 513 articles and language for 507 because GKG metadata lacks those fields.
- Every record is marked `unverified_news_mention`; no company, country, or severity attribution is fabricated.

### Disruption Monitoring (restricted)
- CC BY-NC 4.0. Kept outside the main graph and training data.
- The KG has 6,222 companies and 16,640 edges; scenarios and benchmark items are authored synthetic evaluation material.
- Do not use in commercial deliverables without permission.

## Labels

- `training/risk_labels.csv`: **all 130,020 nodes are `not_available`**. There are no verified company-level risk labels in the acquired sources.
- `training/delay_labels.csv`: 155,488 SynDelay rows and 1,000 Smart Logistics rows. SynDelay semantics are unverified.
- Restricted benchmark labels are available only under `datasets/evaluation/`.

## Recommendations

1. Do not train a company risk model until verified risk labels are collected.
2. Perform entity resolution only where identifiers are non-anonymised; never merge IC-SPLC/AnonNet IDs with named companies.
3. Obtain written permission for SynDelay and AnonNet before redistribution or commercial use.
4. Confirm Smart Logistics provenance and licence.
5. Treat GDELT as a weak signal; require event confirmation before claiming a disruption.
6. Expand the GDELT window and retry the rate-limited keyword with backoff.
7. Document or validate SynDelay class semantics before reporting delay severity.
8. Handle class imbalance and missing embeddings explicitly in GNN training.
9. Keep restricted CC BY-NC material out of production and commercial artefacts.
