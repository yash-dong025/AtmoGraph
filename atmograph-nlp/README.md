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
| 2 | News ingestion (mock data + RSS) | ⏳ Next |
| 3 | spaCy NER & entity extraction | Pending |
| 4 | Disruption + severity classification | Pending |
| 5 | Entity normalization & matching | Pending |
| 6 | Risk intelligence | Pending |
| 7 | Full FastAPI endpoint set | Pending |
| 8 | Member 1 backend / Neo4j integration | Pending |
| 9 | GNN payload adapter | Pending |
| 10 | Full test suite & documentation | Pending |

Only Phase 1 code exists right now. Empty packages are placeholders for the
components above; no stub logic has been written ahead of its phase.

---

## 2. Architecture

```
News (RSS / manual / API)
        |
        v
Ingestion  ->  Cleaning  ->  spaCy NER + entity extraction
                                      |
                                      v
                        Disruption + severity classification
                                      |
                                      v
                   Entity matching -> Risk intelligence
                                      |
                    +-----------------+------------------+
                    v                                    v
        Member 1 backend / Neo4j                Member 3 GNN payload
                    |                                    |
                    +-----------> React dashboard <-------+
```

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
│   ├── main.py                  # FastAPI app + health endpoint (Phase 1)
│   ├── config.py                # Pydantic settings loaded from .env
│   ├── ingestion/               # rss_reader, news_fetcher, article_cleaner (Phase 2)
│   ├── nlp/                     # preprocessing, ner, entity_extractor (Phase 3)
│   ├── classification/          # disruption + severity classifiers (Phase 4)
│   ├── intelligence/            # entity matcher + risk analyzer (Phases 5-6)
│   ├── integration/             # backend/Neo4j clients + GNN adapter (Phases 8-9)
│   ├── schemas/                 # Pydantic API models
│   └── utils/                   # logger + helpers
├── data/                        # sample_news.json and processed store
├── scripts/verify_env.py        # environment smoke check
├── tests/                       # pytest suite
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
- Internet access for the first dependency/model download.

> **This machine:** the system-wide Python registrations are broken (the
> 32-bit interpreter cannot install spaCy). A 64-bit Python 3.13 was installed
> per-user with `uv` and the project virtual environment was created from it.
> The commands below work as-is; activation is not required.

---

## 5. Setup (Windows PowerShell)

```powershell
cd C:\Users\Vidhi\OneDrive\Desktop\AtmoGraph\atmograph-nlp

# 1. Virtual environment (already created; recreate only if missing)
& "$env:USERPROFILE\.local\bin\uv.exe" venv --python 3.13 --seed .venv

# 2. Dependencies
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Local configuration (never commit .env)
Copy-Item .env.example .env

# 4. Verify the environment
.\.venv\Scripts\python.exe scripts\verify_env.py

# 5. Run the API
.\.venv\Scripts\python.exe run.py --reload
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

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health | ConvertTo-Json -Depth 5
```

Expected shape:

```json
{
  "status": "ok",
  "service": "AtmoGraph NLP & News Intelligence",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2026-11-21T15:30:00+00:00",
  "components": [
    { "name": "spacy", "available": true, "required": true, "version": "3.8.x" }
  ],
  "features": {
    "transformer_classifier": false,
    "graph_sync": false,
    "spacy_model": "en_core_web_sm"
  }
}
```

Interactive docs: <http://127.0.0.1:8000/docs>

### Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

---

## 7. API endpoints

| Method | Path | Phase |
|--------|------|-------|
| GET | `/api/health` | ✅ 1 |
| POST | `/api/news/ingest` | 7 |
| POST | `/api/news/process` | 7 |
| POST | `/api/news/classify` | 7 |
| GET | `/api/news` | 7 |
| GET | `/api/news/{article_id}` | 7 |
| GET | `/api/disruptions` | 7 |
| GET | `/api/disruptions/{event_id}` | 7 |
| POST | `/api/news/process-batch` | 7 |

---

## 8. Configuration

Copy `.env.example` to `.env`. Every variable uses the `ATMOGRAPH_` prefix.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ATMOGRAPH_ENVIRONMENT` | `development` | `development`, `testing`, `production` |
| `ATMOGRAPH_PORT` | `8000` | API port |
| `ATMOGRAPH_LOG_LEVEL` | `INFO` | Console/file log level |
| `ATMOGRAPH_SPACY_MODEL` | `en_core_web_sm` | spaCy pipeline |
| `ATMOGRAPH_ENABLE_TRANSFORMER` | `false` | Enable optional transformer classifier |
| `ATMOGRAPH_RSS_FEEDS` | BBC + Supply Chain Dive | Comma-separated RSS URLs |
| `ATMOGRAPH_GRAPH_SYNC_ENABLED` | `false` | Push events to Member 1's backend |
| `ATMOGRAPH_BACKEND_BASE_URL` | `http://127.0.0.1:8001` | Member 1 API base URL |

Secrets (`ATMOGRAPH_BACKEND_API_KEY`, `ATMOGRAPH_NEO4J_PASSWORD`) are optional,
read as `SecretStr`, and must never be committed.

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

## 9. Logging

Logs go to the console and `logs/atmograph-nlp.log` (5 MB rotation, 3 backups).
Feed failures, backend retries and classifier fallbacks are logged with context.

---

## 10. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No module named spacy` | Install dependencies with the project interpreter (`.\.venv\Scripts\python.exe`) |
| spaCy install fails / no `thinc` wheel | You are on 32-bit Python; use 64-bit 3.10-3.13 |
| `Activate.ps1 cannot be loaded` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` or call the interpreter directly |
| Health shows `degraded` | Check the `components` array for the missing required package |
| Port already in use | `run.py --port 8010` or stop the other process |
| RSS feed timeout | Expected; the ingestion layer logs and skips unavailable feeds |
