from app.database.neo4j_connection import db

CONSTRAINTS = [
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT disruption_id IF NOT EXISTS FOR (d:Disruption) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT prediction_id IF NOT EXISTS FOR (p:Prediction) REQUIRE p.id IS UNIQUE",
]
INDEXES = [
    "CREATE INDEX entity_kind IF NOT EXISTS FOR (n:Entity) ON (n.kind)",
    "CREATE INDEX entity_risk IF NOT EXISTS FOR (n:Entity) ON (n.risk_score)",
]


def initialize_schema():
    for statement in CONSTRAINTS + INDEXES:
        db.query(statement, read=False)
