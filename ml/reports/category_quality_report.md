# Country and Industry Category-Quality Report

Generated: `2026-11-13T04:04:26.035657+00:00`  
Dataset version: `atmo_graph_v1_015d0c924af0`  
Input: `datasets/processed/graph/nodes.csv`  
Total nodes: **130,020**

## Root cause

- Source file: `datasets/raw/ic_splc/extracted_csv/ic_splc_nodes.csv`
- Source columns: `country`, `industry_1st`, `category_1st`
- IC-SPLC categorical values are anonymized frequency-ranked codes. The publisher does not provide a mapping to real countries or industries. preprocess_ic_splc.py preserves them, build_graph_dataset.py combines them with AnonNet/SAMPOQ nodes, and the dashboard previously aggregated them without source context.
- Source metadata: `datasets/raw/ic_splc/mendeley_metadata.json`

## Summary

| Field | Verified nodes | Anonymized nodes | Missing nodes | Unknown nodes | Verified distinct | Anonymized distinct |
|---|---:|---:|---:|---:|---:|---:|
| Country | 5,755 | 25,122 | 99,143 | 0 | 102 | 179 |
| Industry | 0 | 10,565 | 119,455 | 0 | 0 | 22 |

## Dataset-wise category distribution

| Source | Total | Country verified | Country anonymized | Country missing | Industry verified | Industry anonymized | Industry missing |
|---|---:|---:|---:|---:|---:|---:|---:|
| AnonNet | 7,245 | 5,755 | 0 | 1,490 | 0 | 0 | 7,245 |
| IC-SPLC | 109,403 | 0 | 25,122 | 84,281 | 0 | 10,565 | 98,838 |
| SAMPOQ | 13,372 | 0 | 0 | 13,372 | 0 | 0 | 13,372 |

## Verified countries

Source: AnonNet `location`. Names are preserved exactly as provided.

| Label | Raw value | Source | Nodes | Share of type |
|---|---|---|---:|---:|
| Japan | `Japan` | AnonNet | 1,099 | 19.1% |
| China | `China` | AnonNet | 1,086 | 18.9% |
| United States | `United States` | AnonNet | 720 | 12.5% |
| South Korea | `South Korea` | AnonNet | 481 | 8.4% |
| Taiwan | `Taiwan` | AnonNet | 300 | 5.2% |
| India | `India` | AnonNet | 249 | 4.3% |
| Germany | `Germany` | AnonNet | 154 | 2.7% |
| Indonesia | `Indonesia` | AnonNet | 129 | 2.2% |
| United Kingdom | `United Kingdom` | AnonNet | 119 | 2.1% |
| Australia | `Australia` | AnonNet | 106 | 1.8% |
| Canada | `Canada` | AnonNet | 99 | 1.7% |
| Spain | `Spain` | AnonNet | 97 | 1.7% |
| Vietnam | `Vietnam` | AnonNet | 83 | 1.4% |
| Russian Federation | `Russian Federation` | AnonNet | 78 | 1.4% |
| France | `France` | AnonNet | 62 | 1.1% |
| Cayman Islands | `Cayman Islands` | AnonNet | 61 | 1.1% |
| Netherlands | `Netherlands` | AnonNet | 46 | 0.8% |
| Malaysia | `Malaysia` | AnonNet | 45 | 0.8% |
| Italy | `Italy` | AnonNet | 39 | 0.7% |
| Mexico | `Mexico` | AnonNet | 39 | 0.7% |

## Anonymized country codes

Source: IC-SPLC `country`. These are anonymized codes, not countries.

| Label | Raw value | Source | Nodes | Share of type |
|---|---|---|---:|---:|
| Country Code C | `C` | IC-SPLC | 6,117 | 24.3% |
| Country Code B | `B` | IC-SPLC | 5,158 | 20.5% |
| Country Code D | `D` | IC-SPLC | 2,835 | 11.3% |
| Country Code E | `E` | IC-SPLC | 1,865 | 7.4% |
| Country Code F | `F` | IC-SPLC | 1,865 | 7.4% |
| Country Code H | `H` | IC-SPLC | 933 | 3.7% |
| Country Code G | `G` | IC-SPLC | 595 | 2.4% |
| Country Code L | `L` | IC-SPLC | 465 | 1.9% |
| Country Code P | `P` | IC-SPLC | 404 | 1.6% |
| Country Code N | `N` | IC-SPLC | 370 | 1.5% |
| Country Code I | `I` | IC-SPLC | 359 | 1.4% |
| Country Code Y | `Y` | IC-SPLC | 322 | 1.3% |
| Country Code O | `O` | IC-SPLC | 293 | 1.2% |
| Country Code AE | `AE` | IC-SPLC | 205 | 0.8% |
| Country Code J | `J` | IC-SPLC | 201 | 0.8% |
| Country Code M | `M` | IC-SPLC | 199 | 0.8% |
| Country Code W | `W` | IC-SPLC | 195 | 0.8% |
| Country Code AO | `AO` | IC-SPLC | 155 | 0.6% |
| Country Code S | `S` | IC-SPLC | 146 | 0.6% |
| Country Code T | `T` | IC-SPLC | 128 | 0.5% |

## Verified industries

No source in the current graph provides verified industry names.

_None._

## Anonymized industry codes

Source: IC-SPLC `industry_1st`. These are anonymized codes, not industries.

| Label | Raw value | Source | Nodes | Share of type |
|---|---|---|---:|---:|
| Industry Code A | `A` | IC-SPLC | 3,476 | 32.9% |
| Industry Code C | `C` | IC-SPLC | 2,054 | 19.4% |
| Industry Code B | `B` | IC-SPLC | 1,891 | 17.9% |
| Industry Code D | `D` | IC-SPLC | 974 | 9.2% |
| Industry Code F | `F` | IC-SPLC | 316 | 3.0% |
| Industry Code H | `H` | IC-SPLC | 292 | 2.8% |
| Industry Code E | `E` | IC-SPLC | 283 | 2.7% |
| Industry Code G | `G` | IC-SPLC | 245 | 2.3% |
| Industry Code L | `L` | IC-SPLC | 211 | 2.0% |
| Industry Code K | `K` | IC-SPLC | 183 | 1.7% |
| Industry Code J | `J` | IC-SPLC | 171 | 1.6% |
| Industry Code I | `I` | IC-SPLC | 143 | 1.4% |
| Industry Code M | `M` | IC-SPLC | 99 | 0.9% |
| Industry Code N | `N` | IC-SPLC | 68 | 0.6% |
| Industry Code O | `O` | IC-SPLC | 47 | 0.4% |
| Industry Code P | `P` | IC-SPLC | 35 | 0.3% |
| Industry Code R | `R` | IC-SPLC | 33 | 0.3% |
| Industry Code Q | `Q` | IC-SPLC | 20 | 0.2% |
| Industry Code S | `S` | IC-SPLC | 12 | 0.1% |
| Industry Code T | `T` | IC-SPLC | 8 | 0.1% |

## Missing values by source

- **Country:** AnonNet: 1,490, IC-SPLC: 84,281, SAMPOQ: 13,372
- **Industry:** AnonNet: 7,245, IC-SPLC: 98,838, SAMPOQ: 13,372

## Before the fix

The previous dashboard aggregated all non-empty raw values without source context.

| Rank | Mixed top countries | Count | Raw top industries | Count |
|---:|---|---:|---|---:|
| 1 | `C` | 6,117 | `A` | 3,476 |
| 2 | `B` | 5,158 | `C` | 2,054 |
| 3 | `D` | 2,835 | `B` | 1,891 |
| 4 | `F` | 1,865 | `D` | 974 |
| 5 | `E` | 1,865 | `F` | 316 |
| 6 | `Japan` | 1,099 | `H` | 292 |
| 7 | `China` | 1,086 | `E` | 283 |
| 8 | `H` | 933 | `G` | 245 |
| 9 | `United States` | 720 | `L` | 211 |
| 10 | `G` | 595 | `K` | 183 |
| 11 | `South Korea` | 481 | `J` | 171 |
| 12 | `L` | 465 | `I` | 143 |

## Validation

Overall status: **PASS**

| Check | Status | Expected | Actual |
|---|---|---|---|
| country: category-type counts sum to total nodes | PASS | `130020` | `130020` |
| country: per-source totals match processed nodes.csv | PASS | `{'ic_splc': 109403, 'sampoq': 13372, 'anonnet': 7245}` | `{'anonnet': 7245, 'ic_splc': 109403, 'sampoq': 13372}` |
| country: raw distinct-value counts sum to non-missing nodes | PASS | `30877` | `30877` |
| country: distinct raw values match direct pandas value_counts | PASS | `281` | `281` |
| country: top displayed values match direct raw counts | PASS | `{'Japan': 1099, 'China': 1086, 'United States': 720, 'South Korea': 481, 'Taiwan': 300, 'India': 249, 'Germany': 154, 'Indonesia': 129, 'United Kingdom': 119, 'Australia': 106, 'Canada': 99, 'Spain': 97, 'Vietnam': 83, 'Russian Federation': 78, 'France': 62, 'Cayman Islands': 61, 'Netherlands': 46, 'Malaysia': 45, 'Italy': 39, 'Mexico': 39, 'Singapore': 39, 'Thailand': 39, 'Brazil': 36, 'Chile': 35, 'Hong Kong': 34, 'C': 6117, 'B': 5158, 'D': 2835, 'E': 1865, 'F': 1865, 'H': 933, 'G': 595, 'L': 465, 'P': 404, 'N': 370, 'I': 359, 'Y': 322, 'O': 293, 'AE': 205, 'J': 201, 'M': 199, 'W': 195, 'AO': 155, 'S': 146, 'T': 128, 'AB': 121, 'AH': 119, 'AA': 104, 'AJ': 101, 'AG': 96}` | `{'Japan': 1099, 'China': 1086, 'United States': 720, 'South Korea': 481, 'Taiwan': 300, 'India': 249, 'Germany': 154, 'Indonesia': 129, 'United Kingdom': 119, 'Australia': 106, 'Canada': 99, 'Spain': 97, 'Vietnam': 83, 'Russian Federation': 78, 'France': 62, 'Cayman Islands': 61, 'Netherlands': 46, 'Malaysia': 45, 'Italy': 39, 'Mexico': 39, 'Singapore': 39, 'Thailand': 39, 'Brazil': 36, 'Chile': 35, 'Hong Kong': 34, 'C': 6117, 'B': 5158, 'D': 2835, 'E': 1865, 'F': 1865, 'H': 933, 'G': 595, 'L': 465, 'P': 404, 'N': 370, 'I': 359, 'Y': 322, 'O': 293, 'AE': 205, 'J': 201, 'M': 199, 'W': 195, 'AO': 155, 'S': 146, 'T': 128, 'AB': 121, 'AH': 119, 'AA': 104, 'AJ': 101, 'AG': 96}` |
| industry: category-type counts sum to total nodes | PASS | `130020` | `130020` |
| industry: per-source totals match processed nodes.csv | PASS | `{'ic_splc': 109403, 'sampoq': 13372, 'anonnet': 7245}` | `{'anonnet': 7245, 'ic_splc': 109403, 'sampoq': 13372}` |
| industry: raw distinct-value counts sum to non-missing nodes | PASS | `10565` | `10565` |
| industry: distinct raw values match direct pandas value_counts | PASS | `22` | `22` |
| industry: top displayed values match direct raw counts | PASS | `{'A': 3476, 'C': 2054, 'B': 1891, 'D': 974, 'F': 316, 'H': 292, 'E': 283, 'G': 245, 'L': 211, 'K': 183, 'J': 171, 'I': 143, 'M': 99, 'N': 68, 'O': 47, 'P': 35, 'R': 33, 'Q': 20, 'S': 12, 'T': 8, 'U': 2, 'W': 2}` | `{'A': 3476, 'C': 2054, 'B': 1891, 'D': 974, 'F': 316, 'H': 292, 'E': 283, 'G': 245, 'L': 211, 'K': 183, 'J': 171, 'I': 143, 'M': 99, 'N': 68, 'O': 47, 'P': 35, 'R': 33, 'Q': 20, 'S': 12, 'T': 8, 'U': 2, 'W': 2}` |

Full raw and normalized distinct-value lists are in `category_quality_report.json`.
