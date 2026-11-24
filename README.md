# AtmoGraph — Member 1 (Neo4j Desktop + FastAPI)

**Complete code, not templates.** Member 1 owns graph database, REST APIs, graph queries and integration interfaces. NLP classification, ML training and React UI belong to other members. Dataset is **synthetic**.

## Run using Windows PowerShell

1. In Neo4j Desktop create/start a local DBMS. Check its **Bolt port** and user password. The default database should be `neo4j`.
2. Open this folder in VS Code and run:

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

3. Edit `backend/.env` with the actual Neo4j Desktop password and Bolt URI. **Never commit `.env`.**
4. From the `backend` directory, run:

```powershell
python -m scripts.init_db
python -m scripts.seed_graph
uvicorn app.main:app --reload
```

Both setup scripts print clear status messages and exit with a nonzero code on connection errors. `seed_graph` prints **17 nodes, 16 relationships**, and database totals.

5. Test URLs:
    - http://127.0.0.1:8000/health
    - http://127.0.0.1:8000/docs
    - http://127.0.0.1:8000/api/v1/graph
    - http://127.0.0.1:8000/api/v1/graph/downstream/mumbai_port?max_hops=2

6. Run isolated tests without Neo4j Desktop:

```powershell
pytest -q
```

## In Neo4j Desktop Query

```cypher
MATCH (n:Entity) RETURN n.kind AS kind, count(*) AS total ORDER BY kind;
```

```cypher
MATCH p=(a:Entity {id:'mumbai_port'})-[:SHIPS_TO|DISTRIBUTES_TO*1..2]->(b:Entity) RETURN p LIMIT 25;
```

## Project structure

- `backend/app/database/`: Neo4j connection and Cypher schema.
- `backend/app/models/`: typed REST input models.
- `backend/app/repositories/`: parameterized Cypher CRUD and traversal queries.
- `backend/app/services/`: validation and orchestration.
- `backend/app/api/routes/`: health, graph CRUD, explorer, and team integration endpoints.
- `backend/scripts/`: CLI database initialization and CSV seeding.
- `backend/data/`: synthetic supply-chain CSVs.
- `backend/tests/`: isolated pytest tests.
- `docs/`: graph schema and API contract.

## Integration ownership

Member 2 calls `POST /api/v1/disruptions` with an entity ID and extracted event; Member 3 can consume `GET /api/v1/graph` and store model outputs through `POST /api/v1/predictions`; Member 4 retrieves graph and predictions through these REST interfaces. **This service stores model output; it does not perform ML inference.**

## Caveats

- Designed for local Neo4j Desktop. Authentication/authorization and HTTPS must be added before public deployment.
- Risk scores are example numeric attributes; they are not calibrated real-world probabilities.
- Seed script is repeat-safe but intentionally overwrites sample entities' `risk_score` to zero each time: do not reseed after recording live test events unless you want to reset the demo data.
- Merge into the team's existing project carefully: don't overwrite other members' `main.py` or routes without merging.
