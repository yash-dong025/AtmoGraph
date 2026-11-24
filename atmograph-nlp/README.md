# AtmoGraph - NLP & News Intelligence Module

**Member 2 deliverable** for *AtmoGraph - AI-Powered Global Supply Chain Disruption
Prediction System*.

This service turns unstructured news into structured supply-chain intelligence:
entities, disruption category, severity, event-level risk, graph matches and a
GNN-ready payload. It does **not** train the GNN or own the Neo4j schema.

---

## 1. Current status

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Project setup, configuration, logging, health endpoint | ✅ Complete |
| 2 | News ingestion (mock data + RSS), deduplication, JSON storage, news API | ✅ Complete |
| 3 | spaCy NER, supply-chain entity extraction, normalization, annotation storage, NLP API | ✅ Complete |
| 4 | Disruption + severity classification | ⏳ Next |
| 5 | Graph entity matching / fuzzy resolution | Pending |
| 6 | Risk intelligence | Pending |
| 7 | Full FastAPI endpoint set | Pending |
| 8 | Member 1 backend / Neo4j integration | Pending |
| 9 | GNN payload adapter | Pending |
| 10 | Full test suite & documentation | Pending |

Phase 3 is implemented and covered by **108 passing pytest tests** (65 from
Phases 1-2 plus 43 new Phase 3 tests). Tests use mocked RSS responses and the
locally installed `en_core_web_sm` model; they never require live internet access
or a model download. Disruption classification, graph matching, risk scoring and
Neo4j writes are intentionally **not** implemented yet.

---

## 2. Architecture

```
News (RSS / mock dataset)
        |
        v
Ingestion  ->  Validation + normalization  ->  Duplicate detection
                                                        |
                                                        v
                                              data/articles.json
                                                        |
                                                        v
                         Phase 3: spaCy NER + configurable Matcher rules
                                                        |
                                                        v
                              data/entity_annotations.json
                                                        |
                                                        v
                         Phase 4+: classification -> graph matching -> risk
                                                        |
                                                        v
                                      Member 1 backend / Neo4j, Member 3 GNN
```

**Phase 2 flow**

1. `MockIngestor` loads `data/mock_news.json`; `RssIngestor` fetches configured
   public feeds concurrently with a timeout.
2. Every record becomes an `Article` (Pydantic v2): whitespace/HTML cleanup,
   UTC publication date, normalized URL, stable content hash and article ID.
3. `DeduplicationIndex` checks article ID, normalized URL and content hash.
4. `JsonArticleStorage` appends only new articles using an atomic file replace.
5. FastAPI routes return articles and structured ingestion statistics.

**Phase 3 flow**

1. `POST /api/nlp/extract/{article_id}` loads the stored article and runs
   `en_core_web_sm` once per field (`title`, then `description`).
2. spaCy NER supplies generic labels (`ORG`, `GPE`, `LOC`, `FAC`, `PRODUCT`,
   `PERSON`, `DATE`); spaCy `PhraseMatcher`/`Matcher` rules add supply-chain
   types (ports, facilities, materials, industries, products).
3. Overlapping candidates are resolved deterministically; implausible headline
   fragments are rejected; names are cleaned and aliases are resolved.
4. The complete annotation record is written atomically to
   `data/entity_annotations.json`. Article storage is never modified.
5. `GET /api/nlp/entities/{article_id}`, `GET /api/nlp/stats` and
   `GET /api/nlp/health` expose the stored annotations and service state.

**Boundaries**

- This module generates **event-level** risk only. Multi-hop impact and GNN
  predictions belong to Member 3.
- Graph writes go through Member 1's API. Direct Neo4j access is optional and
  disabled by default.
- Entities are only reported when supported by the article text; missing values
  are omitted. No confidence score is invented. Each annotation records whether
  it came from statistical NER (`spacy`) or a rule (`rule`) and its raw label.
- Entity normalization in Phase 3 is alias-level (for example `U.S.` →
  `United States`). Fuzzy matching against graph nodes belongs to Phase 5.

---

## 3. Project structure

```
atmograph-nlp/
├── app/
│   ├── main.py                     # FastAPI app + health endpoint
│   ├── config.py                   # Pydantic settings loaded from .env
│   ├── api/
│   │   ├── news.py                 # Phase 2 news routes
│   │   └── nlp.py                  # Phase 3 entity extraction routes
│   ├── ingestion/
│   │   ├── feed_config.py          # Configurable public RSS catalog
│   │   ├── mock_ingestor.py        # data/mock_news.json loader
│   │   ├── rss_ingestor.py         # httpx + feedparser fetching
│   │   ├── deduplication.py        # URL / content-hash duplicate index
│   │   └── ingestion_service.py    # Orchestration + statistics
│   ├── nlp/
│   │   ├── taxonomy.py             # Entity types, gazetteers, heuristics
│   │   ├── patterns.py             # Built-in + custom Matcher patterns
│   │   ├── model_loader.py         # Lazy, thread-safe spaCy singleton
│   │   ├── normalizer.py           # Cleaning, aliases, canonical keys
│   │   ├── extractor.py            # NER + rule matching + overlap resolution
│   │   └── service.py              # Extraction orchestration, stats, health
│   ├── schemas/
│   │   ├── article.py              # Normalized Article schema
│   │   ├── ingestion.py            # Ingestion/listing response models
│   │   ├── nlp.py                  # Phase 3 annotation/API models
│   │   └── health.py               # Phase 1 health models
│   ├── storage/
│   │   ├── json_storage.py         # Atomic JSON article store
│   │   └── annotation_storage.py   # Atomic JSON annotation store
│   ├── classification/             # Phase 4 placeholders
│   ├── intelligence/               # Phases 5-6 placeholders
│   ├── integration/                # Phases 8-9 placeholders
│   └── utils/                      # logger + helpers
├── data/
│   ├── mock_news.json              # 22 original mock articles
│   ├── articles.json               # Created automatically at runtime
│   └── entity_annotations.json     # Created automatically at runtime
├── scripts/verify_env.py           # environment smoke check
├── tests/                          # pytest suite (108 tests)
├── .env.example
├── pytest.ini
├── requirements.txt
├── run.py
└── README.md
```

`data/articles.json` and `data/entity_annotations.json` are generated files and
are excluded from Git.

---

## 4. Prerequisites

- **64-bit Python 3.10 - 3.13.** spaCy has no 32-bit Windows wheels.
- PowerShell 5.1 or newer.
- Internet access for the first dependency/model download and for live RSS.
- The spaCy model `en_core_web_sm` must already be installed in the project
  virtual environment. The service never downloads a model at request time.

> The project virtual environment is already configured. Activation is not
> required; call `.\.venv\Scripts\python.exe` directly.

---

## 5. Setup (Windows PowerShell)

```powershell
cd <path-to-project>\atmograph-nlp

# 1. Dependencies (already installed in the existing .venv)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Local configuration (never commit .env)
Copy-Item .env.example .env

# 3. Verify the environment
.\.venv\Scripts\python.exe scripts\verify_env.py

# 4. Run the API on port 8010
.\.venv\Scripts\python.exe run.py --port 8010 --reload
```

Optional activation instead of the `.\.venv\Scripts\python.exe` prefix:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### spaCy model (required for Phase 3)

The model is already installed in the project virtual environment. If it is
missing on another machine, install it once:

```powershell
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

The API loads the model lazily on the first extraction request. If the package is
missing, `/api/nlp/health` reports `degraded`, and extraction endpoints return
`503` with a clear installation message.

---

## 6. Verify the running service

| Resource | URL |
|----------|-----|
| Swagger UI | <http://127.0.0.1:8010/docs> |
| ReDoc | <http://127.0.0.1:8010/redoc> |
| Health check | <http://127.0.0.1:8010/api/health> |
| NLP health | <http://127.0.0.1:8010/api/nlp/health> |
| NLP statistics | <http://127.0.0.1:8010/api/nlp/stats> |
| OpenAPI JSON | <http://127.0.0.1:8010/openapi.json> |

```powershell
Invoke-RestMethod http://127.0.0.1:8010/api/health | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8010/api/nlp/health | ConvertTo-Json -Depth 5
```

The Phase 1 health response is unchanged. It reports dependency availability and
feature flags. The Phase 3 NLP health endpoint additionally reports whether the
spaCy model is installed/loaded, the configured rule-pattern count, and whether
the annotation store is readable.

### Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Current result: **108 passed**. Tests use mocked RSS responses and never require
live internet access. If the pytest cache directory is locked on Windows, add
`-p no:cacheprovider`.

---

## 7. API endpoints

### Phase 1 (preserved)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Service and dependency status |

### Phase 2 (preserved)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/news` | List stored articles with pagination and optional source filter |
| POST | `/api/news/ingest/mock` | Ingest `data/mock_news.json` |
| POST | `/api/news/ingest/rss` | Ingest all configured RSS feeds |
| POST | `/api/news/ingest/all` | Ingest mock + RSS in one run |
| GET | `/api/news/stats` | Storage totals and last-run counters |
| GET | `/api/news/feeds` | Configured feed names and public URLs |

### Phase 3 (new)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/nlp/extract/{article_id}` | Extract and persist entities for one article |
| POST | `/api/nlp/extract/batch` | Extract entities for explicit IDs or a selected page |
| GET | `/api/nlp/entities/{article_id}` | Read the stored annotation for one article |
| GET | `/api/nlp/stats` | Annotation coverage and entity-type counts |
| GET | `/api/nlp/health` | Model, pattern and annotation-store health |

### `POST /api/nlp/extract/{article_id}`

Runs spaCy NER plus the configured supply-chain rules, normalizes names and
upserts the annotation in `data/entity_annotations.json`. Calling it again
replaces the previous annotation. Use `?overwrite=false` to return an existing
annotation unchanged.

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8010/api/nlp/extract/mock-001 |
    ConvertTo-Json -Depth 8
```

Example response (abridged):

```json
{
  "article_id": "mock-001",
  "title": "Typhoon Halts Container Operations at Kaohsiung Terminal",
  "source_type": "mock",
  "model_name": "en_core_web_sm",
  "extracted_at": "2026-11-24T10:00:00Z",
  "entity_count": 5,
  "entities": [
    {
      "article_id": "mock-001",
      "text": "Kaohsiung Terminal",
      "normalized_name": "Kaohsiung Terminal",
      "entity_type": "FACILITY",
      "source_field": "title",
      "start_char": 38,
      "end_char": 56,
      "extraction_method": "rule",
      "raw_label": "ATMO_FACILITY"
    },
    {
      "article_id": "mock-001",
      "text": "Taiwan",
      "normalized_name": "Taiwan",
      "entity_type": "COUNTRY",
      "source_field": "description",
      "start_char": 97,
      "end_char": 103,
      "extraction_method": "spacy",
      "raw_label": "GPE"
    }
  ]
}
```

### `POST /api/nlp/extract/batch`

Supply explicit `article_ids`, or omit them to select a page from article storage
using `limit`, `offset` and `source_type`. Explicit IDs take precedence over
pagination. The request size is capped by `ATMOGRAPH_MAX_BATCH_SIZE` (default
50). Existing annotations are replaced unless `overwrite` is false.

```powershell
$body = @{ limit = 3; offset = 0; overwrite = $true } | ConvertTo-Json
Invoke-RestMethod -Method Post http://127.0.0.1:8010/api/nlp/extract/batch `
    -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 6
```

```powershell
$body = @{ article_ids = @("mock-001", "mock-002") } | ConvertTo-Json
Invoke-RestMethod -Method Post http://127.0.0.1:8010/api/nlp/extract/batch `
    -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 6
```

Response shape:

```json
{
  "requested": 3,
  "processed": 2,
  "skipped": 0,
  "not_found": 1,
  "failed": 0,
  "results": [
    {"article_id": "mock-001", "status": "extracted", "entity_count": 5, "annotation": {}},
    {"article_id": "mock-002", "status": "extracted", "entity_count": 6, "annotation": {}},
    {"article_id": "missing", "status": "not_found", "entity_count": 0, "annotation": null}
  ],
  "started_at": "2026-11-24T10:00:00Z",
  "finished_at": "2026-11-24T10:00:02Z"
}
```

`status` is one of `extracted`, `skipped`, `not_found` or `failed`.

### `GET /api/nlp/entities/{article_id}`

Returns the stored annotation. Returns `404` when the article itself does not
exist or when it has not been extracted yet.

```powershell
Invoke-RestMethod http://127.0.0.1:8010/api/nlp/entities/mock-001 |
    ConvertTo-Json -Depth 8
```

### `GET /api/nlp/stats`

```json
{
  "total_articles": 22,
  "annotated_articles": 2,
  "articles_with_entities": 2,
  "total_entities": 11,
  "entity_counts": [
    {"entity_type": "COMPANY", "count": 0},
    {"entity_type": "ORGANIZATION", "count": 0},
    {"entity_type": "COUNTRY", "count": 2},
    {"entity_type": "LOCATION", "count": 2},
    {"entity_type": "PORT", "count": 1},
    {"entity_type": "FACILITY", "count": 4},
    {"entity_type": "PRODUCT", "count": 1},
    {"entity_type": "MATERIAL", "count": 1},
    {"entity_type": "INDUSTRY", "count": 0},
    {"entity_type": "PERSON", "count": 0},
    {"entity_type": "DATE", "count": 0}
  ],
  "average_entities_per_article": 5.5,
  "model_name": "en_core_web_sm",
  "last_extracted_at": "2026-11-24T10:00:02Z"
}
```

Counts include zero values so dashboards can render a stable set of keys. The
example assumes the 22-article mock dataset; a live store may also contain RSS
articles, in which case `total_articles` will be larger.

### `GET /api/nlp/health`

```json
{
  "status": "ok",
  "model_name": "en_core_web_sm",
  "model_available": true,
  "model_loaded": true,
  "model_version": "3.8.0",
  "pipeline_components": [
    "tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"
  ],
  "pattern_count": 190,
  "annotation_store_path": "C:\\...\\data\\entity_annotations.json",
  "annotation_store_readable": true,
  "timestamp": "2026-11-24T10:00:00Z"
}
```

The endpoint does **not** load the model just to answer the check. `status` is
`degraded` when the model package is missing, the optional pattern file is
invalid, or the annotation store cannot be read.

### Error responses

| Status | When |
|--------|------|
| `404` | Article ID not found, or the article has no stored annotation yet |
| `422` | Invalid query/body shape, empty article ID, or batch larger than `ATMOGRAPH_MAX_BATCH_SIZE` |
| `500` | Article or annotation store is unreadable/contains invalid JSON (the file is left untouched) |
| `503` | spaCy model is missing or the optional pattern file is invalid |

---

## 8. Entity taxonomy and extraction pipeline

### Taxonomy

| Type | Primary source | Example surface text |
|------|----------------|----------------------|
| `COMPANY` | spaCy `ORG` without organisation cues; GPE company-context heuristic | `TSMC`, `Foxconn`, `Toyota` |
| `ORGANIZATION` | spaCy `ORG` with organisation cues + known-organisation rules | `World Trade Organization`, `WTO`, `United Nations` |
| `COUNTRY` | spaCy `GPE` + country gazetteer | `Taiwan`, `China`, `United States` (`U.S.` normalized) |
| `LOCATION` | spaCy `GPE`/`LOC` + location gazetteer | `Geneva`, `Zhengzhou`, `Red Sea` |
| `PORT` | Rules + port gazetteer | `Port of Shanghai`, `Port of Los Angeles` |
| `FACILITY` | Rules + spaCy `FAC` | `Kaohsiung Terminal`, `container terminal`, `semiconductor plant` |
| `PRODUCT` | Rules + spaCy `PRODUCT` | `container`, `battery`, `machine tool` |
| `MATERIAL` | Rules | `steel`, `aluminum`, `lithium`, `cobalt`, `resin` |
| `INDUSTRY` | Rules | `semiconductor`, `automotive`, `logistics`, `textile` |
| `PERSON` | spaCy `PERSON` with headline-fragment filter | `Mei-Ling Chen` |
| `DATE` | spaCy `DATE` | `24 November 2026`, `two-week` |

### Normalization

Every mention keeps its original `text`, exact `start_char`/`end_char` offsets and
`source_field`. A separate `normalized_name` is produced by:

1. Unicode NFKC normalization, curly-quote cleanup and whitespace collapsing.
2. Removing leading articles (`The`, `A`, `An`) and possessive endings (`China's`).
3. Resolving known aliases: `U.S.`/`USA` → `United States`, `UK` → `United
   Kingdom`, `UAE` → `United Arab Emirates`, `Aluminium` → `Aluminum`, `WTO` →
   `World Trade Organization`, and similar country/material/organisation aliases.
4. Applying canonical display names for known gazetteer terms (`steel` → `Steel`,
   `semiconductor` → `Semiconductor`, `port of shanghai` → `Port of Shanghai`).

A punctuation/case-folded canonical key is used for duplicate checks, so
`U.S.`, `USA` and `United States` compare equal.

### Overlap resolution and precision guards

- Rule matches receive a length bonus, so a domain phrase such as
  `Kaohsiung Terminal` can override a generic `ORG` label while a long
  organisation name is not fragmented by a short keyword.
- Candidates are selected greedily, longest-and-rule-first, with no overlapping
  output spans.
- Headline fragments such as `Fire Forces Temporary Closure of Auto Parts
  Stamping Plant` are rejected as `ORG`/`PERSON` candidates.
- Weather names (`Typhoon Meranti`) are not reported as products or people.
- Entities shorter than two characters, punctuation-only spans and duplicate
  mentions are dropped.
- No confidence score is fabricated. `extraction_method` (`spacy` or `rule`) and
  `raw_label` identify provenance.

### Custom patterns

Extra phrase or token patterns can be loaded from a JSON file:

```json
{
  "patterns": [
    {"label": "PORT", "pattern": "Port of My Town"},
    {"label": "MATERIAL", "pattern": [{"LOWER": "cork"}]},
    {"label": "FACILITY", "pattern": [{"LOWER": "orbital"}, {"LOWER": "depot"}]}
  ]
}
```

Set `ATMOGRAPH_NLP_PATTERNS_PATH=data/entity_patterns.json` (or another path).
Labels must be one of the taxonomy types; `ATMO_`-prefixed labels are also
accepted. A missing or malformed file is reported by `/api/nlp/health` and makes
extraction return `503` rather than silently ignoring the configuration.

---

## 9. Configuration

Copy `.env.example` to `.env`. Every variable uses the `ATMOGRAPH_` prefix.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ATMOGRAPH_ENVIRONMENT` | `development` | `development`, `testing`, `production` |
| `ATMOGRAPH_PORT` | `8000` | API port (use `--port 8010` to override) |
| `ATMOGRAPH_LOG_LEVEL` | `INFO` | Console/file log level |
| `ATMOGRAPH_MOCK_NEWS_PATH` | `data/mock_news.json` | Mock article dataset |
| `ATMOGRAPH_ARTICLES_STORE_PATH` | `data/articles.json` | Local normalized article store |
| `ATMOGRAPH_ENTITY_ANNOTATIONS_PATH` | `data/entity_annotations.json` | Local entity-annotation store |
| `ATMOGRAPH_NLP_PATTERNS_PATH` | unset | Optional custom phrase/token pattern JSON |
| `ATMOGRAPH_SPACY_MODEL` | `en_core_web_sm` | spaCy pipeline used for NER |
| `ATMOGRAPH_RSS_FEEDS` | 9 public feeds | Comma-separated URLs or `Name\|URL` entries |
| `ATMOGRAPH_RSS_TIMEOUT_SECONDS` | `10` | Per-feed request timeout |
| `ATMOGRAPH_RSS_MAX_ITEMS_PER_FEED` | `25` | Maximum entries read from each feed |
| `ATMOGRAPH_NEWS_USER_AGENT` | `AtmoGraph-NLP/0.1 ...` | HTTP user agent |
| `ATMOGRAPH_ARTICLE_MAX_CHARS` | `20000` | Maximum stored plain-text description |
| `ATMOGRAPH_MAX_BATCH_SIZE` | `50` | Maximum articles per NLP batch request |
| `ATMOGRAPH_ENABLE_TRANSFORMER` | `false` | Optional transformer classifier (Phase 4) |
| `ATMOGRAPH_GRAPH_SYNC_ENABLED` | `false` | Push events to Member 1's backend |
| `ATMOGRAPH_BACKEND_BASE_URL` | `http://127.0.0.1:8001` | Member 1 API base URL |

Secrets (`ATMOGRAPH_BACKEND_API_KEY`, `ATMOGRAPH_NEO4J_PASSWORD`) are optional,
read as `SecretStr`, and must never be committed.

### Feed configuration examples

```dotenv
# Plain URLs (names come from the built-in catalog or hostname)
ATMOGRAPH_RSS_FEEDS=https://feeds.bbci.co.uk/news/business/rss.xml,https://gcaptain.com/feed/

# Optional explicit display name
ATMOGRAPH_RSS_FEEDS=My Logistics Feed|https://example.com/logistics.xml
```

Invalid entries, duplicate URLs and unavailable feeds are ignored/logged without
stopping the run. Feed URLs containing `user:password@` are sanitized before
they are used or exposed.

### Temporary integration contract

Until Member 1 publishes the final endpoints, the planned calls are isolated in
`app/integration/backend_client.py` (Phase 8):

| Method | Planned path | Purpose |
|--------|--------------|---------|
| `POST` | `/api/graph/entities/match` | Resolve normalized entities to node IDs |
| `POST` | `/api/graph/disruptions` | Upsert disruption event + article reference |
| `PATCH` | `/api/graph/entities/{id}/risk` | Update node risk properties |

This table is a placeholder for team coordination, **not** a finalized schema.

---

## 10. Annotation storage and data guarantees

Annotations are stored separately from articles so the Phase 2 ingestion contract
stays stable and annotations can be regenerated without touching source data.

`data/entity_annotations.json` is a JSON array of article records:

```json
[
  {
    "article_id": "mock-001",
    "title": "Typhoon Halts Container Operations at Kaohsiung Terminal",
    "source_type": "mock",
    "model_name": "en_core_web_sm",
    "extracted_at": "2026-11-24T10:00:00Z",
    "entity_count": 2,
    "entities": [
      {
        "article_id": "mock-001",
        "text": "Kaohsiung Terminal",
        "normalized_name": "Kaohsiung Terminal",
        "entity_type": "FACILITY",
        "source_field": "title",
        "start_char": 38,
        "end_char": 56,
        "extraction_method": "rule",
        "raw_label": "ATMO_FACILITY"
      }
    ]
  }
]
```

Guarantees:

- **Atomic writes:** records are written to a temporary file in the same
  directory, flushed and moved into place with `os.replace`.
- **Corrupt files are never overwritten:** invalid JSON raises a clear `500` and
  the original file stays unchanged.
- **Invalid records are skipped on read** and preserved verbatim when another
  article is upserted.
- **Upsert is idempotent per article:** re-extraction replaces the record instead
  of appending duplicates.
- **Empty/missing files** are treated as an empty store.
- **Separate interface:** `EntityAnnotationStorage` allows a database backend to
  replace `JsonEntityAnnotationStorage` later without changing API code.
- **Generated files are gitignored:** `data/articles.json` and
  `data/entity_annotations.json`.

---

## 11. Duplicate detection and article storage guarantees

- **URL duplicates:** URLs are lowercased, `www.` and default ports are removed,
  fragments and tracking parameters (`utm_*`, `fbclid`, …) are dropped, query
  parameters are sorted, and trailing slashes are removed.
- **Content duplicates:** SHA-256 of the case-folded, whitespace-normalized title
  and description. This catches the same story across different feeds and between
  RSS and mock data.
- **ID duplicates:** explicitly supplied IDs (for example `mock-001`) are
  preserved and checked as a third key.
- **Atomic writes:** the JSON file is written to a temporary file in the same
  directory, flushed and moved into place with `os.replace`.
- **Corrupt files:** invalid JSON raises a clear `500` error and the original file
  is never overwritten. Empty files are treated as an empty store.
- **Invalid records:** records that fail schema validation are skipped and counted;
  invalid records already present in the JSON file are preserved when new
  articles are appended.

The mock dataset includes a `category` label on each record as Phase 4 test
metadata (for example `labor_strike` or `port_congestion`). It is intentionally
not part of the Phase 2 `Article` schema; classification is a Phase 4 concern.

The `ArticleStorage` interface allows a database backend to replace
`JsonArticleStorage` later without changing ingestion or API code.

---

## 12. Logging

Logs go to the console and `logs/atmograph-nlp.log` (5 MB rotation, 3 backups).
Each feed fetch logs its name/URL, success/failure, fetched/valid/invalid counts,
and every ingestion run logs aggregate statistics. Feed timeouts and malformed
responses are warnings, not crashes.

Phase 3 logs model loading, per-article entity counts, batch summaries and
annotation-store failures. The model is loaded once per process; inference is
serialized so concurrent API requests do not share an unsafe pipeline state.

---

## 13. Known limitations (Phase 3)

- `en_core_web_sm` is a small model. It can confuse `ORG`/`GPE`/`PERSON` labels
  (for example a city detected as a company). Heuristics and headline filters
  reduce but do not eliminate these errors.
- Port, facility, material, industry and product coverage comes from English
  gazetteers and token patterns. New domains require custom patterns through
  `ATMOGRAPH_NLP_PATTERNS_PATH`.
- Normalization is alias-level only. Fuzzy matching, entity linking and graph
  node resolution are Phase 5 work.
- No confidence scores are produced. `extraction_method` and `raw_label` show
  provenance instead of an invented probability.
- No coreference resolution or cross-article entity merging. Each article is
  processed independently.
- The JSON stores are designed for a single process. Concurrent writers across
  multiple worker processes are not coordinated; use a database backend before
  scaling horizontally.
- Annotations are replaced on re-extraction. There is no historical model-version
  archive yet.
- RSS entries are summarized as supplied by the feed. Full article-body scraping
  is not part of Phase 2/3.
- No disruption classification, severity scoring, risk scoring, Neo4j writes or
  GNN payload generation is performed yet.

---

## 14. Suggested next steps (Phase 4+)

1. Add disruption-category and severity classification (rule baseline first,
   optional transformer second) using the `category` labels in the mock dataset.
2. Match normalized entities against Member 1's graph entities (Phase 5) and
   publish a stable entity/event contract.
3. Add event-level risk scoring and the risk-intelligence endpoints.
4. Integrate Member 1's backend/Neo4j contract and the GNN payload adapter.
5. Build the dashboard views for entities, disruptions, risk and graph matches.
6. Expand the test suite with golden annotation fixtures and regression checks for
   the taxonomy/gazetteers.

---

## 15. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No module named spacy` | Install dependencies with `.\.venv\Scripts\python.exe` |
| spaCy install fails / no `thinc` wheel | Use 64-bit Python 3.10-3.13 |
| `Activate.ps1 cannot be loaded` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` or call the interpreter directly |
| `503` on `/api/nlp/extract/...` | Install `en_core_web_sm` or fix `ATMOGRAPH_NLP_PATTERNS_PATH` |
| `/api/nlp/entities/...` returns `404` | Run `POST /api/nlp/extract/{article_id}` first |
| NLP health is `degraded` | Check `model_available`, `annotation_store_readable` and `pattern_count` in `/api/nlp/health` |
| Health shows `degraded` (Phase 1) | Check the `components` array for the missing required package |
| Port already in use | `run.py --port 8010` or stop the other process |
| RSS feed timeout / 404 | Expected for unavailable public feeds; the run continues and reports `feeds_failed` |
| `500` mentioning invalid JSON | Inspect `data/articles.json` or `data/entity_annotations.json`; the service deliberately refuses to overwrite them |
| Custom patterns ignored / `503` | Validate the JSON labels and token-pattern shape; see section 8 |
| pytest cache permission error | Add `-p no:cacheprovider` |
