"""Seed the local database using synthetic CSV supply-chain data.

Run from backend/: python -m scripts.seed_graph
Idempotent: rerunning updates the sample nodes and relationships.
"""
import csv
from pathlib import Path
from neo4j.exceptions import Neo4jError
from app.database.neo4j_connection import db
from app.database.schema import initialize_schema
from app.repositories.graph_repository import create_relation

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def seed():
    initialize_schema()
    with (DATA_DIR / "nodes.csv").open(encoding="utf-8-sig", newline="") as f:
        nodes = list(csv.DictReader(f))
    with (DATA_DIR / "relationships.csv").open(encoding="utf-8-sig", newline="") as f:
        links = list(csv.DictReader(f))

    for item in nodes:
        db.query("""
            MERGE (n:Entity {id:$id})
            SET n.name=$name, n.kind=$kind, n.city=$city,
                n.country=$country, n.risk_score=$risk_score
        """, {**item, "risk_score":float(item["risk_score"])}, read=False)

    for item in links:
        result = create_relation({**item, "transit_days":float(item["transit_days"])})
        if result is None:
            raise RuntimeError(f"Relationship references missing entity: {item}")

    counts = db.query("""
       MATCH (n:Entity)
       WITH count(n) AS entities
       OPTIONAL MATCH (:Entity)-[r]->(:Entity)
       WHERE type(r) IN ['SUPPLIES_TO','SHIPS_TO','DISTRIBUTES_TO']
       RETURN entities, count(r) AS relationships
    """)
    print(f"SUCCESS: Seeded {len(nodes)} sample nodes and {len(links)} links.", flush=True)
    print(f"Database totals: {counts[0]['entities']} entities, {counts[0]['relationships']} supply-chain links.", flush=True)


def main():
    print("Connecting to Neo4j Desktop and seeding data...", flush=True)
    try:
        db.verify_connection()
        seed()
    except (Neo4jError, OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(1) from exc
    finally:
        db.close()


if __name__ == "__main__":
    main()
