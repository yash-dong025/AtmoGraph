from neo4j import GraphDatabase, RoutingControl
from app.config import get_settings


class Neo4jConnection:
    def __init__(self):
        self.driver = None

    def connect(self):
        if self.driver is None:
            config = get_settings()
            if not config.NEO4J_PASSWORD:
                raise RuntimeError("Set NEO4J_PASSWORD in backend/.env")
            self.driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
            )
        return self.driver

    def verify_connection(self):
        self.connect().verify_connectivity()

    def query(self, cypher: str, params: dict | None = None, *, read: bool = True) -> list[dict]:
        records, _, _ = self.connect().execute_query(
            cypher,
            parameters_=params or {},
            database_=get_settings().NEO4J_DATABASE,
            routing_=RoutingControl.READ if read else RoutingControl.WRITE,
        )
        return [record.data() for record in records]

    def close(self):
        if self.driver is not None:
            self.driver.close()
            self.driver = None


db = Neo4jConnection()
