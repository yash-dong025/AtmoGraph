from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.database.neo4j_connection import db
from app.api.routes import health, graph, explorer, integrations


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    db.close()


settings = get_settings()
app = FastAPI(
    title=settings.APP_NAME,
    description="Member 1 graph database and integration APIs for AtmoGraph",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)
app.include_router(health.router)
app.include_router(graph.router)
app.include_router(explorer.router)
app.include_router(integrations.router)
