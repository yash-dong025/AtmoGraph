"""FastAPI routers for the AtmoGraph NLP service."""

from app.api.news import router as news_router

__all__ = ["news_router"]
