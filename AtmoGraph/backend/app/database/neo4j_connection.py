
from neo4j import GraphDatabase
from app.config import settings


class Neo4jConnection:

    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(
                settings.NEO4J_USER,
                settings.NEO4J_PASSWORD
            )
        )

    def verify_connection(self):
        self.driver.verify_connectivity()
        return True

    def execute_query(self, query, parameters=None):
        records, summary, keys = (
            self.driver.execute_query(
                query,
                parameters_=parameters or {},
                database_=settings.NEO4J_DATABASE
            )
        )

        return [record.data() for record in records]

    def close(self):
        self.driver.close()


neo4j_db = Neo4jConnection()
