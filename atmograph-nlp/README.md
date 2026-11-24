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
| 3 | spaCy NER & entity extraction | ⏳ Next |
| 4 | Disruption + severity classification | Pending |
| 5 | Entity normalization & matching | Pending |
| 6 | Risk intelligence | Pending |
| 7 | Full FastAPI endpoint set | Pending |
| 8 | Member 1 backend / Neo4j integration | Pending |
| 9 | GNN payload adapter | Pending |
| 10 | Full test suite & documentation | Pending |

Phase 2 is implemented and covered by 65 passing pytest tests. NLP entity
extraction and disruption classification are intentionally **not** implemented
yet.

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
                                   Phase 3+: spaCy NER -> classification
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

**Boundaries**

- This module generates **event-level** risk only. Multi-hop impact and GNN
  predictions belong to Member 3.
- Graph writes go through Member 1's API. Direct Neo4j access is optional and
  disabled by default.
- Entities are only reported when supported by the article text; missing values
  are `null` / `unknown` with low confidence.

---

## 3. Project structure

```
atmograph-nlp/
├── app/
│   ├── main.py                     # FastAPI app + health endpoint
│   ├── config.py                   # Pydantic settings loaded from .env
│   ├── api/
│   │   └── news.py                 # Phase 2 news routes
│   ├── ingestion/
│   │   ├── feed_config.py          # Configurable public RSS catalog
│   │   ├── mock_ingestor.py        # data/mock_news.json loader
│   │   ├── rss_ingestor.py         # httpx + feedparser fetching
│   │   ├── deduplication.py        # URL / content-hash duplicate index
│   │   └── ingestion_service.py    # Orchestration + statistics
│   ├── schemas/
│   │   ├── article.py              # Normalized Article schema
│   │   ├── ingestion.py            # Ingestion/listing response models
│   │   └── health.py               # Phase 1 health models
│   ├── storage/
│   │   └── json_storage.py         # Atomic JSON article store
│   ├── nlp/                        # Phase 3 placeholders
│   ├── classification/             # Phase 4 placeholders
│   ├── intelligence/               # Phases 5-6 placeholders
│   ├── integration/                # Phases 8-9 placeholders
│   └── utils/                      # logger + helpers
├── data/
│   ├── mock_news.json              # 22 original mock articles
│   └── articles.json               # Created automatically at runtime
├── scripts/verify_env.py           # environment smoke check
├── tests/                          # pytest suite (65 tests)
├── .env.example
├── pytest.ini
├── requirements.txt
├── run.py
└── README.md
```

---

## 4. Prerequisites

- **64-bit Python 3.10 - 3.13.** spaCy has no 32-bit Windows wheels.
- PowerShell 5.1 or newer.
- Internet access for the first dependency/model download and for live RSS.

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

### Download the spaCy model (required in Phase 3)

```powershell
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

---

## 6. Verify the running service

| Resource | URL |
|----------|-----|
| Swagger UI | <http://127.0.0.1:8010/docs> |
| ReDoc | <http://127.0.0.1:8010/redoc> |
| Health check | <http://127.0.0.1:8010/api/health> |
| OpenAPI JSON | <http://127.0.0.1:8010/openapi.json> |

```powershell
Invoke-RestMethod http://127.0.0.1:8010/api/health | ConvertTo-Json -Depth 5
```

The Phase 1 health response is unchanged. It reports dependency availability and
feature flags; news ingestion adds no required dependency beyond the packages
already listed.

### Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Current result: **65 passed**. Tests use mocked RSS responses and never require
live internet access. If the pytest cache directory is locked on Windows, add
`-p no:cacheprovider`.

---

## 7. API endpoints

### Phase 1 (preserved)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Service and dependency status |

### Phase 2 (new)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/news` | List stored articles with pagination and optional source filter |
| POST | `/api/news/ingest/mock` | Ingest `data/mock_news.json` |
| POST | `/api/news/ingest/rss` | Ingest all configured RSS feeds |
| POST | `/api/news/ingest/all` | Ingest mock + RSS in one run |
| GET | `/api/news/stats` | Storage totals and last-run counters |
| GET | `/api/news/feeds` | Configured feed names and public URLs |

### `GET /api/news`

Query parameters:

| Parameter | Default | Rules |
|-----------|---------|-------|
| `limit` | `20` | `1..200` |
| `offset` | `0` | `>= 0` |
| `source_type` | none | `mock` or `rss` |

Example:

```powershell
Invoke-RestMethod "http://127.0.0.1:8010/api/news?limit=5&offset=0&source_type=mock" |
    ConvertTo-Json -Depth 6
```

Response shape:

```json
{
  "items": [
    {
      "article_id": "mock-001",
      "title": "Typhoon Halts Container Operations at Kaohsiung Terminal",
      "description": "Terminal operators suspended vessel berthing ...",
      "source": "Port Watch Digest",
      "url": "https://example.com/atmograph/mock/typhoon-kaohsiung-terminal",
      "published_at": "2026-11-18T06:40:00Z",
      "author": "Mei-Ling Chen",
      "source_type": "mock",
      "content_hash": "…64 hex characters…",
      "ingested_at": "2026-11-22T10:00:00Z"
    }
  ],
  "total": 22,
  "limit": 5,
  "offset": 0
}
```

### `POST /api/news/ingest/mock`

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8010/api/news/ingest/mock |
    ConvertTo-Json -Depth 5
```

```json
{
  "status": "success",
  "total_fetched": 22,
  "total_saved": 22,
  "duplicates_skipped": 0,
  "invalid_articles": 0,
  "feeds_successful": 0,
  "feeds_failed": 0,
  "source": "mock",
  "errors": [],
  "started_at": "2026-11-22T10:00:00Z",
  "finished_at": "2026-11-22T10:00:01Z"
}
```

Calling it a second time returns `total_saved: 0` and
`duplicates_skipped: 22`.

### `POST /api/news/ingest/rss`

Fetches every configured feed concurrently. A failed feed is logged and counted
but does not abort the remaining feeds. Typical response:

```json
{
  "status": "partial",
  "total_fetched": 74,
  "total_saved": 61,
  "duplicates_skipped": 8,
  "invalid_articles": 5,
  "feeds_successful": 7,
  "feeds_failed": 2,
  "source": "rss",
  "errors": ["Example Feed: HTTP 404 from https://example.com/feed"],
  "started_at": "2026-11-22T10:00:00Z",
  "finished_at": "2026-11-22T10:00:09Z"
}
```

### `POST /api/news/ingest/all`

Runs both loaders and returns combined counters. Mock and RSS records are
deduplicated against each other as well as against stored articles.

### `GET /api/news/stats`

```json
{
  "total_articles": 24,
  "rss_articles": 2,
  "mock_articles": 22,
  "duplicates_skipped": 0,
  "invalid_articles": 0,
  "feeds_successful": 1,
  "feeds_failed": 0,
  "last_ingestion_source": "all",
  "last_ingestion_at": "2026-11-22T10:00:09Z"
}
```

`duplicates_skipped`, `invalid_articles`, `feeds_successful` and `feeds_failed`
describe the **most recent run in the current process**. Storage totals are
persisted in `data/articles.json`; last-run counters reset when the service
restarts.

### `GET /api/news/feeds`

```json
{
  "count": 9,
  "feeds": [
    {
      "name": "BBC Business",
      "url": "https://feeds.bbci.co.uk/news/business/rss.xml",
      "category": "trade"
    },
    {
      "name": "BBC World",
      "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
      "category": "natural_disaster"
    }
  ]
}
```

Feed URLs are sanitized so embedded credentials are never returned. The module
uses public RSS feeds only; no paid APIs or API keys are required.

### Error responses

| Status | When |
|--------|------|
| `422` | Invalid query parameter or request shape |
| `500` | Article store is unreadable or contains invalid JSON (the file is left untouched) |

---

## 8. Configuration

Copy `.env.example` to `.env`. Every variable uses the `ATMOGRAPH_` prefix.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ATMOGRAPH_ENVIRONMENT` | `development` | `development`, `testing`, `production` |
| `ATMOGRAPH_PORT` | `8000` | API port (use `--port 8010` to override) |
| `ATMOGRAPH_LOG_LEVEL` | `INFO` | Console/file log level |
| `ATMOGRAPH_MOCK_NEWS_PATH` | `data/mock_news.json` | Mock article dataset |
| `ATMOGRAPH_ARTICLES_STORE_PATH` | `data/articles.json` | Local normalized article store |
| `ATMOGRAPH_RSS_FEEDS` | 9 public feeds | Comma-separated URLs or `Name\|URL` entries |
| `ATMOGRAPH_RSS_TIMEOUT_SECONDS` | `10` | Per-feed request timeout |
| `ATMOGRAPH_RSS_MAX_ITEMS_PER_FEED` | `25` | Maximum entries read from each feed |
| `ATMOGRAPH_NEWS_USER_AGENT` | `AtmoGraph-NLP/0.1 ...` | HTTP user agent |
| `ATMOGRAPH_ARTICLE_MAX_CHARS` | `20000` | Maximum stored plain-text description |
| `ATMOGRAPH_SPACY_MODEL` | `en_core_web_sm` | spaCy pipeline (Phase 3) |
| `ATMOGRAPH_ENABLE_TRANSFORMER` | `false` | Optional transformer classifier |
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

## 9. Duplicate detection and storage guarantees

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

The mock dataset includes a `category` label on each record as Phase 3 test
metadata (for example `labor_strike` or `port_congestion`). It is intentionally
not part of the Phase 2 `Article` schema; classification is a Phase 3 concern.

The `ArticleStorage` interface allows a database backend to replace
`JsonArticleStorage` later without changing ingestion or API code.

---

## 10. Logging

Logs go to the console and `logs/atmograph-nlp.log` (5 MB rotation, 3 backups).
Each feed fetch logs its name/URL, success/failure, fetched/valid/invalid counts,
and every ingestion run logs aggregate statistics. Feed timeouts and malformed
responses are warnings, not crashes.

---

## 11. Known limitations (Phase 2)

- Last-run counters (`duplicates_skipped`, feed counts) are in memory and reset
  when the process restarts. Stored article counts are persistent.
- The JSON store is designed for a single process. Concurrent writers across
  multiple worker processes are not coordinated; use a database backend before
  scaling horizontally.
- RSS entries are summarized as supplied by the feed. Full article-body scraping
  is not part of Phase 2.
- Content-hash deduplication requires both title and description to match after
  normalization. Near-duplicate rewrites are intentionally not merged.
- A feed that returns a valid but empty channel is reported as a failed feed so
  statistics highlight it; this can be revisited if empty feeds become normal.
- No entity extraction, disruption classification, risk scoring or Neo4j writes
  are performed yet.

---

## 12. Suggested next steps (Phase 3)

1. Load `en_core_web_sm` once at startup and add a spaCy NER service.
2. Extract organizations, locations, products/materials, ports and dates from
   stored articles.
3. Add entity normalization and matching against Member 1's graph entities.
4. Extend the article model with NLP annotations in a separate Phase 3 schema so
   the Phase 2 ingestion contract stays stable.
5. Add `/api/news/{article_id}/entities` or a batch processing endpoint.
6. Publish a stable article/event contract for Member 3's GNN and Member 4's
   dashboard.

---

## 13. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No module named spacy` | Install dependencies with `.\.venv\Scripts\python.exe` |
| spaCy install fails / no `thinc` wheel | Use 64-bit Python 3.10-3.13 |
| `Activate.ps1 cannot be loaded` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` or call the interpreter directly |
| Health shows `degraded` | Check the `components` array for the missing required package |
| Port already in use | `run.py --port 8010` or stop the other process |
| RSS feed timeout / 404 | Expected for unavailable public feeds; the run continues and reports `feeds_failed` |
| `500` on `/api/news` mentioning invalid JSON | Inspect `data/articles.json`; the service deliberately refuses to overwrite it |
| pytest cache permission error | Add `-p no:cacheprovider` |
