# AtmoGraph Member 1 REST contract

Base: `http://127.0.0.1:8000`; OpenAPI: `/docs`.

| Method | Path | Used by |
|---|---|---|
| GET | `/health` | All |
| POST / GET | `/api/v1/entities` | Graph CRUD |
| GET / PATCH / DELETE | `/api/v1/entities/{id}` | Graph CRUD |
| POST / DELETE | `/api/v1/relationships` | Graph CRUD |
| GET | `/api/v1/graph?limit=500` | Frontend + GNN |
| GET | `/api/v1/graph/downstream/{id}?max_hops=3` | Frontend + NLP |
| POST | `/api/v1/disruptions` | NLP team |
| POST | `/api/v1/predictions` | GNN team |
| GET | `/api/v1/predictions/{id}` | Frontend |

## Examples

POST `/api/v1/disruptions`:
```json
{"id":"flood-mumbai-2026-11-12", "entity_id":"mumbai_port", "kind":"FLOOD", "severity":"HIGH", "description":"Heavy rainfall disrupted Mumbai Port", "risk_score":49.4}
```

POST `/api/v1/predictions`:
```json
{"id":"prediction-nov12-bengaluru-30d", "entity_id":"bengaluru_warehouse", "horizon_days":30, "predicted_delay_days":8, "risk_score":55, "model_name":"GraphSAGE-demo"}
```
These example predictions are illustrative **only**; the backend does not run or verify GraphSAGE.

GET `/api/v1/graph` returns `{ "nodes": [...], "edges": [...] }`. Edge direction is from source to target.

GET `/api/v1/graph/downstream/mumbai_port` returns reachable targets, shortest hop counts, and one shortest directed path by ID. This is **reachability**, not an ML risk prediction.

### Security note
This is a local development API and has no authentication. Do not expose it publicly until authentication, authorization, rate limiting, audit logging, and production configuration are added.
