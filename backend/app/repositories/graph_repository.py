from app.database.neo4j_connection import db

# Every query is defined server-side. User inputs are passed as Cypher parameters.
# Relationships are enumerated rather than interpolated from arbitrary client text.


def create_entity(data: dict):
    result = db.query("""
        CREATE (n:Entity {
          id:$id, name:$name, kind:$kind, country:$country,
          city:$city, risk_score:$risk_score
        }) RETURN properties(n) AS entity
    """, data, read=False)
    return result[0]["entity"]


def list_entities(kind: str | None, limit: int, offset: int):
    return db.query("""
        MATCH (n:Entity) WHERE $kind IS NULL OR n.kind = $kind
        RETURN properties(n) AS entity ORDER BY n.id
        SKIP $offset LIMIT $limit
    """, {"kind": kind, "limit": limit, "offset": offset})


def get_entity(entity_id: str):
    rows = db.query("MATCH (n:Entity {id:$id}) RETURN properties(n) AS entity", {"id": entity_id})
    return rows[0]["entity"] if rows else None


def update_entity(entity_id: str, changes: dict):
    rows = db.query("""
        MATCH (n:Entity {id:$id}) SET n += $changes
        RETURN properties(n) AS entity
    """, {"id": entity_id, "changes": changes}, read=False)
    return rows[0]["entity"] if rows else None


def delete_entity(entity_id: str):
    rows = db.query("""
        MATCH (n:Entity {id:$id})
        WITH n, properties(n) AS old
        DETACH DELETE n
        RETURN old AS entity
    """, {"id": entity_id}, read=False)
    return rows[0]["entity"] if rows else None


def create_relation(data: dict):
    # Static, explicit mapping prevents Cypher injection via relationship types.
    allowed = {"SUPPLIES_TO", "SHIPS_TO", "DISTRIBUTES_TO"}
    relation = data["kind"]
    if relation not in allowed:
        raise ValueError("Unsupported relationship type")
    cypher = f"""
        MATCH (a:Entity {{id:$source_id}}), (b:Entity {{id:$target_id}})
        MERGE (a)-[r:{relation}]->(b)
        SET r.transit_days = $transit_days
        RETURN a.id AS source_id, b.id AS target_id,
               type(r) AS kind, r.transit_days AS transit_days
    """
    rows = db.query(cypher, data, read=False)
    return rows[0] if rows else None


def delete_relation(source_id: str, target_id: str, kind: str):
    rows = db.query("""
        MATCH (a:Entity {id:$source})-[r]->(b:Entity {id:$target})
        WHERE type(r) = $kind
        WITH r, a.id AS source_id, b.id AS target_id, type(r) AS relation_type
        DELETE r
        RETURN source_id, target_id, relation_type AS kind
    """, {"source": source_id, "target": target_id, "kind": kind}, read=False)
    return rows[0] if rows else None


def graph_export(limit: int = 500):
    nodes = db.query("MATCH (n:Entity) RETURN properties(n) AS node ORDER BY n.id LIMIT $limit", {"limit": limit})
    ids = [x["node"]["id"] for x in nodes]
    edges = db.query("""
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE a.id IN $ids AND b.id IN $ids
          AND type(r) IN ['SUPPLIES_TO','SHIPS_TO','DISTRIBUTES_TO']
        RETURN a.id AS source, b.id AS target, type(r) AS kind,
               r.transit_days AS transit_days
        ORDER BY source, target
    """, {"ids": ids})
    return {"nodes": [x["node"] for x in nodes], "edges": edges}


def downstream(entity_id: str, max_hops: int, limit: int):
    return db.query("""
        MATCH p=(start:Entity {id:$id})-[:SUPPLIES_TO|SHIPS_TO|DISTRIBUTES_TO*1..6]->(dest:Entity)
        WHERE length(p) <= $hops
          AND all(n IN nodes(p) WHERE single(m IN nodes(p) WHERE m = n))
        WITH dest, p ORDER BY length(p), [n IN nodes(p) | n.id]
        WITH dest, collect(p)[0] AS shortest
        RETURN dest.id AS entity_id, dest.name AS name,
               length(shortest) AS hops,
               [n IN nodes(shortest) | n.id] AS path
        ORDER BY hops, entity_id LIMIT $limit
    """, {"id": entity_id, "hops": max_hops, "limit": limit})


def record_disruption(data: dict):
    return db.query("""
        MATCH (e:Entity {id:$entity_id})
        MERGE (d:Disruption {id:$id})
        SET d.kind=$kind, d.severity=$severity,
          d.description=$description, d.observed_at=$observed_at,
          d.risk_score=$risk_score
        WITH d, e
        OPTIONAL MATCH (d)-[old:AFFECTS]->(:Entity)
        DELETE old
        MERGE (d)-[:AFFECTS]->(e)
        SET e.risk_score=d.risk_score
        RETURN properties(d) AS disruption, e.id AS entity_id
    """, data, read=False)


def record_prediction(data: dict):
    return db.query("""
        MATCH (e:Entity {id:$entity_id})
        MERGE (p:Prediction {id:$id})
        SET p.horizon_days=$horizon_days,
          p.predicted_delay_days=$predicted_delay_days,
          p.risk_score=$risk_score, p.model_name=$model_name,
          p.generated_at=$generated_at
        WITH p, e
        OPTIONAL MATCH (p)-[old:PREDICTS_FOR]->(:Entity)
        DELETE old
        MERGE (p)-[:PREDICTS_FOR]->(e)
        RETURN properties(p) AS prediction, e.id AS entity_id
    """, data, read=False)


def predictions(entity_id: str):
    return db.query("""
        MATCH (p:Prediction)-[:PREDICTS_FOR]->(e:Entity {id:$id})
        RETURN properties(p) AS prediction ORDER BY p.generated_at DESC
    """, {"id": entity_id})
