"""Initialize Neo4j Desktop constraints and indexes.

Run from backend/: python -m scripts.init_db
"""
from neo4j.exceptions import Neo4jError
from app.database.neo4j_connection import db
from app.database.schema import initialize_schema


def main():
    print("[1/2] Connecting to Neo4j Desktop...", flush=True)
    try:
        db.verify_connection()
        print("[2/2] Creating constraints and indexes...", flush=True)
        initialize_schema()
        print("SUCCESS: Neo4j schema initialized.", flush=True)
    except (Neo4jError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(1) from exc
    finally:
        db.close()


if __name__ == "__main__":
    main()
