"""End-to-end API tests for all Phase 2 news endpoints (no live internet)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.ingestion.ingestion_service import IngestionService, get_ingestion_service
from app.ingestion.mock_ingestor import MockIngestor
from app.ingestion.rss_ingestor import RssIngestor
from app.main import app
from app.schemas.ingestion import FeedConfig
from app.storage.json_storage import JsonArticleStorage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOCK_DATASET = PROJECT_ROOT / "data" / "mock_news.json"

RSS_TWO_ITEMS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>API Test Feed</title>
    <item>
      <title>Bridge closure delays regional truck freight</title>
      <link>https://news.example.com/bridge-closure</link>
      <description>Heavy vehicles are being rerouted for two weeks.</description>
      <pubDate>Wed, 04 Nov 2026 10:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Factory maintenance lengthens component lead times</title>
      <link>https://news.example.com/factory-maintenance</link>
      <description>A planned shutdown will last one extra week.</description>
      <pubDate>Thu, 05 Nov 2026 11:30:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


def _rss_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path in {"/rss", "/rss-copy"}:
        return httpx.Response(200, content=RSS_TWO_ITEMS.encode("utf-8"))
    return httpx.Response(404, text="not found")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="testing",
        rss_feeds="",
        articles_store_path=tmp_path / "articles.json",
        mock_news_path=MOCK_DATASET,
        rss_timeout_seconds=2,
    )


def _service(
    settings: Settings,
    feeds: list[FeedConfig] | None = None,
) -> IngestionService:
    return IngestionService(
        storage=JsonArticleStorage(settings.articles_store_path),
        settings=settings,
        mock_ingestor=MockIngestor(settings=settings),
        rss_ingestor=RssIngestor(
            settings=settings,
            feeds=feeds if feeds is not None else [FeedConfig(name="API Test Feed", url="https://feeds.test/rss")],
            transport=httpx.MockTransport(_rss_handler),
        ),
    )


@pytest.fixture
def api(tmp_path: Path) -> Iterator[tuple[TestClient, IngestionService, Settings]]:
    settings = _settings(tmp_path)
    service = _service(settings)
    app.dependency_overrides[get_ingestion_service] = lambda: service
    try:
        with TestClient(app) as client:
            yield client, service, settings
    finally:
        app.dependency_overrides.clear()


def test_news_endpoints_are_documented_in_openapi(api: tuple[TestClient, IngestionService, Settings]) -> None:
    client, _, _ = api
    schema = client.get("/openapi.json").json()

    expected = {
        "/api/news",
        "/api/news/stats",
        "/api/news/feeds",
        "/api/news/ingest/mock",
        "/api/news/ingest/rss",
        "/api/news/ingest/all",
    }
    assert expected.issubset(set(schema["paths"]))

    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "swagger" in docs.text.lower()


def test_health_endpoint_is_preserved(api: tuple[TestClient, IngestionService, Settings]) -> None:
    client, _, _ = api
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded"}


def test_empty_article_listing(api: tuple[TestClient, IngestionService, Settings]) -> None:
    client, _, _ = api
    response = client.get("/api/news")

    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert body["total"] == 0
    assert body["limit"] == 20
    assert body["offset"] == 0


def test_mock_ingestion_and_duplicate_handling(api: tuple[TestClient, IngestionService, Settings]) -> None:
    client, _, _ = api

    first = client.post("/api/news/ingest/mock")
    assert first.status_code == 200
    body = first.json()
    assert body["status"] == "success"
    assert body["source"] == "mock"
    assert body["total_fetched"] >= 15
    assert body["total_saved"] == body["total_fetched"]
    assert body["duplicates_skipped"] == 0
    assert body["invalid_articles"] == 0
    assert body["feeds_successful"] == 0
    assert body["feeds_failed"] == 0
    assert body["errors"] == []

    second = client.post("/api/news/ingest/mock")
    assert second.status_code == 200
    duplicate_body = second.json()
    assert duplicate_body["total_saved"] == 0
    assert duplicate_body["duplicates_skipped"] == body["total_fetched"]

    listing = client.get("/api/news", params={"limit": 200}).json()
    assert listing["total"] == body["total_fetched"]
    assert len(listing["items"]) == body["total_fetched"]


def test_pagination_and_source_filtering(api: tuple[TestClient, IngestionService, Settings]) -> None:
    client, _, _ = api
    client.post("/api/news/ingest/mock")

    first_page = client.get("/api/news", params={"limit": 5, "offset": 0}).json()
    second_page = client.get("/api/news", params={"limit": 5, "offset": 5}).json()
    assert len(first_page["items"]) == 5
    assert len(second_page["items"]) == 5
    assert first_page["total"] == second_page["total"] >= 15
    assert first_page["items"][0]["article_id"] != second_page["items"][0]["article_id"]

    mock_only = client.get("/api/news", params={"source_type": "mock"}).json()
    rss_only = client.get("/api/news", params={"source_type": "rss"}).json()
    assert mock_only["total"] >= 15
    assert all(item["source_type"] == "mock" for item in mock_only["items"])
    assert rss_only["total"] == 0

    assert client.get("/api/news", params={"source_type": "api"}).status_code == 422
    assert client.get("/api/news", params={"limit": 0}).status_code == 422
    assert client.get("/api/news", params={"offset": -1}).status_code == 422


def test_rss_ingestion_and_stats(api: tuple[TestClient, IngestionService, Settings]) -> None:
    client, _, _ = api

    first = client.post("/api/news/ingest/rss")
    assert first.status_code == 200
    body = first.json()
    assert body["status"] == "success"
    assert body["source"] == "rss"
    assert body["total_fetched"] == 2
    assert body["total_saved"] == 2
    assert body["feeds_successful"] == 1
    assert body["feeds_failed"] == 0

    second = client.post("/api/news/ingest/rss").json()
    assert second["total_saved"] == 0
    assert second["duplicates_skipped"] == 2

    stats = client.get("/api/news/stats")
    assert stats.status_code == 200
    stats_body = stats.json()
    assert stats_body["total_articles"] == 2
    assert stats_body["rss_articles"] == 2
    assert stats_body["mock_articles"] == 0
    assert stats_body["duplicates_skipped"] == 2
    assert stats_body["feeds_successful"] == 1
    assert stats_body["feeds_failed"] == 0
    assert stats_body["last_ingestion_source"] == "rss"
    assert stats_body["last_ingestion_at"] is not None


def test_duplicates_are_detected_across_two_rss_feeds(
    api: tuple[TestClient, IngestionService, Settings],
) -> None:
    client, _, settings = api
    feeds = [
        FeedConfig(name="Feed A", url="https://feeds.test/rss"),
        FeedConfig(name="Feed B", url="https://feeds.test/rss-copy"),
    ]
    app.dependency_overrides[get_ingestion_service] = lambda: _service(settings, feeds=feeds)

    response = client.post("/api/news/ingest/rss")
    assert response.status_code == 200
    body = response.json()
    assert body["total_fetched"] == 4
    assert body["total_saved"] == 2
    assert body["duplicates_skipped"] == 2
    assert body["feeds_successful"] == 2
    assert body["feeds_failed"] == 0


def test_ingest_all_combines_sources(api: tuple[TestClient, IngestionService, Settings]) -> None:
    client, _, _ = api

    response = client.post("/api/news/ingest/all")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["source"] == "all"
    assert body["total_fetched"] == body["total_saved"] >= 17
    assert body["feeds_successful"] == 1
    assert body["feeds_failed"] == 0

    repeated = client.post("/api/news/ingest/all").json()
    assert repeated["total_saved"] == 0
    assert repeated["duplicates_skipped"] == body["total_fetched"]

    stats = client.get("/api/news/stats").json()
    assert stats["total_articles"] == body["total_fetched"]
    assert stats["mock_articles"] >= 15
    assert stats["rss_articles"] == 2
    assert stats["duplicates_skipped"] == body["total_fetched"]


def test_feeds_endpoint_returns_names_and_urls(api: tuple[TestClient, IngestionService, Settings]) -> None:
    client, _, _ = api
    response = client.get("/api/news/feeds")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["feeds"] == [
        {"name": "API Test Feed", "url": "https://feeds.test/rss", "category": "general"}
    ]
    assert "secret" not in json.dumps(body).lower()


def test_failed_feed_is_reported_in_stats(api: tuple[TestClient, IngestionService, Settings]) -> None:
    client, _, settings = api
    feeds = [
        FeedConfig(name="Good Feed", url="https://feeds.test/rss"),
        FeedConfig(name="Bad Feed", url="https://feeds.test/broken"),
    ]
    service = _service(settings, feeds=feeds)
    app.dependency_overrides[get_ingestion_service] = lambda: service

    response = client.post("/api/news/ingest/rss")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partial"
    assert body["feeds_successful"] == 1
    assert body["feeds_failed"] == 1
    assert body["total_saved"] == 2
    assert body["errors"]

    stats = client.get("/api/news/stats").json()
    assert stats["feeds_successful"] == 1
    assert stats["feeds_failed"] == 1


def test_corrupt_store_returns_500_without_destroying_it(
    api: tuple[TestClient, IngestionService, Settings],
) -> None:
    client, _, settings = api
    corrupt = "{ definitely not json"
    settings.articles_store_path.write_text(corrupt, encoding="utf-8")

    response = client.get("/api/news")
    assert response.status_code == 500
    assert "invalid JSON" in response.json()["detail"]
    assert settings.articles_store_path.read_text(encoding="utf-8") == corrupt

    ingest = client.post("/api/news/ingest/mock")
    assert ingest.status_code == 500
    assert settings.articles_store_path.read_text(encoding="utf-8") == corrupt
