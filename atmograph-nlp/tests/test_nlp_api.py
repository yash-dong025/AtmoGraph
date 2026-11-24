"""End-to-end API tests for the Phase 3 NLP endpoints (no live internet)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.ingestion.mock_ingestor import MockIngestor
from app.main import app
from app.nlp.service import NlpService, get_nlp_service
from app.storage.annotation_storage import JsonEntityAnnotationStorage
from app.storage.json_storage import JsonArticleStorage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOCK_DATASET = PROJECT_ROOT / "data" / "mock_news.json"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="testing",
        rss_feeds="",
        articles_store_path=tmp_path / "articles.json",
        entity_annotations_path=tmp_path / "entity_annotations.json",
        mock_news_path=MOCK_DATASET,
        max_batch_size=50,
    )


def _service(settings: Settings) -> NlpService:
    articles = MockIngestor(settings).load().articles
    article_storage = JsonArticleStorage(settings.articles_store_path)
    article_storage.add_articles(articles)
    return NlpService(
        article_storage=article_storage,
        annotation_storage=JsonEntityAnnotationStorage(settings.entity_annotations_path),
        settings=settings,
    )


@pytest.fixture
def api(tmp_path: Path) -> Iterator[tuple[TestClient, NlpService, Settings]]:
    settings = _settings(tmp_path)
    service = _service(settings)
    app.dependency_overrides[get_nlp_service] = lambda: service
    try:
        with TestClient(app) as client:
            yield client, service, settings
    finally:
        app.dependency_overrides.clear()


def _stored_article(service: NlpService, article_id: str):
    return next(
        article for article in service._article_storage.load() if article.article_id == article_id
    )


def test_nlp_endpoints_are_documented(api: tuple[TestClient, NlpService, Settings]) -> None:
    client, _, _ = api
    schema = client.get("/openapi.json").json()

    expected = {
        "/api/nlp/health",
        "/api/nlp/stats",
        "/api/nlp/extract/{article_id}",
        "/api/nlp/extract/batch",
        "/api/nlp/entities/{article_id}",
    }
    assert expected.issubset(set(schema["paths"]))

    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "swagger" in docs.text.lower()


def test_health_reports_model_and_annotation_store(
    api: tuple[TestClient, NlpService, Settings],
) -> None:
    client, _, _ = api
    response = client.get("/api/nlp/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["model_name"] == "en_core_web_sm"
    assert body["model_available"] is True
    assert body["annotation_store_readable"] is True
    assert body["pattern_count"] > 0
    assert body["timestamp"]


def test_extract_get_and_exact_offsets(api: tuple[TestClient, NlpService, Settings]) -> None:
    client, service, settings = api
    article = _stored_article(service, "mock-001")

    response = client.post("/api/nlp/extract/mock-001")
    assert response.status_code == 200
    body = response.json()
    assert body["article_id"] == "mock-001"
    assert body["title"] == article.title
    assert body["source_type"] == "mock"
    assert body["model_name"] == "en_core_web_sm"
    assert body["entity_count"] == len(body["entities"]) > 0

    for entity in body["entities"]:
        assert entity["article_id"] == "mock-001"
        assert entity["entity_type"] in {
            "COMPANY",
            "ORGANIZATION",
            "COUNTRY",
            "LOCATION",
            "PORT",
            "FACILITY",
            "PRODUCT",
            "MATERIAL",
            "INDUSTRY",
            "PERSON",
            "DATE",
        }
        source_text = article.title if entity["source_field"] == "title" else article.description
        assert entity["text"] == source_text[entity["start_char"] : entity["end_char"]]

    stored = client.get("/api/nlp/entities/mock-001")
    assert stored.status_code == 200
    assert stored.json()["entity_count"] == body["entity_count"]
    assert settings.entity_annotations_path.exists()
    assert "mock-001" in settings.entity_annotations_path.read_text(encoding="utf-8")


def test_re_extraction_replaces_one_annotation(
    api: tuple[TestClient, NlpService, Settings],
) -> None:
    client, service, _ = api

    first = client.post("/api/nlp/extract/mock-001").json()
    second = client.post("/api/nlp/extract/mock-001").json()

    assert first["entity_count"] == second["entity_count"]
    assert service._annotation_storage.count() == 1


def test_overwrite_false_reuses_existing_annotation(
    api: tuple[TestClient, NlpService, Settings],
) -> None:
    client, service, _ = api

    first = client.post("/api/nlp/extract/mock-001").json()
    second = client.post("/api/nlp/extract/mock-001", params={"overwrite": "false"}).json()

    assert second["extracted_at"] == first["extracted_at"]
    assert service._annotation_storage.count() == 1


def test_unknown_article_returns_404(api: tuple[TestClient, NlpService, Settings]) -> None:
    client, _, _ = api

    assert client.post("/api/nlp/extract/does-not-exist").status_code == 404
    assert client.get("/api/nlp/entities/does-not-exist").status_code == 404


def test_entities_404_before_extraction(api: tuple[TestClient, NlpService, Settings]) -> None:
    client, _, _ = api
    response = client.get("/api/nlp/entities/mock-002")

    assert response.status_code == 404
    assert "No entity annotation" in response.json()["detail"]


def test_batch_explicit_ids(api: tuple[TestClient, NlpService, Settings]) -> None:
    client, _, _ = api
    response = client.post(
        "/api/nlp/extract/batch",
        json={"article_ids": ["mock-001", "mock-002", "missing"], "overwrite": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["requested"] == 3
    assert body["processed"] == 2
    assert body["not_found"] == 1
    assert body["failed"] == 0
    assert [result["status"] for result in body["results"]] == [
        "extracted",
        "extracted",
        "not_found",
    ]
    assert client.get("/api/nlp/entities/mock-001").status_code == 200
    assert client.get("/api/nlp/entities/mock-002").status_code == 200


def test_batch_pagination_and_limit(api: tuple[TestClient, NlpService, Settings]) -> None:
    client, service, _ = api
    stored_ids = [article.article_id for article in service._article_storage.load()[:3]]

    response = client.post("/api/nlp/extract/batch", json={"limit": 3, "offset": 0})
    assert response.status_code == 200
    body = response.json()
    assert body["requested"] == 3
    assert body["processed"] == 3
    assert [result["article_id"] for result in body["results"]] == stored_ids

    too_large = client.post("/api/nlp/extract/batch", json={"limit": 51})
    assert too_large.status_code == 422
    assert "configured maximum" in too_large.json()["detail"]

    assert client.post("/api/nlp/extract/batch", json={"limit": 0}).status_code == 422


def test_batch_overwrite_false_skips_existing(
    api: tuple[TestClient, NlpService, Settings],
) -> None:
    client, _, _ = api
    first = client.post("/api/nlp/extract/mock-001").json()

    response = client.post(
        "/api/nlp/extract/batch",
        json={"article_ids": ["mock-001"], "overwrite": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["skipped"] == 1
    assert body["processed"] == 0
    assert body["results"][0]["status"] == "skipped"
    assert body["results"][0]["annotation"]["extracted_at"] == first["extracted_at"]


def test_batch_empty_article_id_returns_422(
    api: tuple[TestClient, NlpService, Settings],
) -> None:
    client, _, _ = api
    response = client.post("/api/nlp/extract/batch", json={"article_ids": [""]})

    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


def test_stats_before_and_after_extraction(
    api: tuple[TestClient, NlpService, Settings],
) -> None:
    client, _, _ = api

    empty = client.get("/api/nlp/stats")
    assert empty.status_code == 200
    empty_body = empty.json()
    assert empty_body["total_articles"] == 22
    assert empty_body["annotated_articles"] == 0
    assert empty_body["total_entities"] == 0
    assert empty_body["last_extracted_at"] is None
    assert len(empty_body["entity_counts"]) == 11
    assert all(item["count"] == 0 for item in empty_body["entity_counts"])

    client.post("/api/nlp/extract/mock-001")
    client.post("/api/nlp/extract/mock-002")

    response = client.get("/api/nlp/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["total_articles"] == 22
    assert body["annotated_articles"] == 2
    assert body["articles_with_entities"] == 2
    assert body["total_entities"] > 0
    assert sum(item["count"] for item in body["entity_counts"]) == body["total_entities"]
    assert body["average_entities_per_article"] == pytest.approx(
        body["total_entities"] / body["annotated_articles"]
    )
    assert body["last_extracted_at"] is not None
    assert body["model_name"] == "en_core_web_sm"


def test_corrupt_annotation_store_returns_500_without_destroying_it(
    api: tuple[TestClient, NlpService, Settings],
) -> None:
    client, _, settings = api
    corrupt = "{ definitely not json"
    settings.entity_annotations_path.write_text(corrupt, encoding="utf-8")

    entities = client.get("/api/nlp/entities/mock-001")
    assert entities.status_code == 500
    assert "invalid JSON" in entities.json()["detail"]

    extraction = client.post("/api/nlp/extract/mock-001")
    assert extraction.status_code == 500
    assert "invalid JSON" in extraction.json()["detail"]

    assert settings.entity_annotations_path.read_text(encoding="utf-8") == corrupt


def test_health_is_degraded_when_annotation_store_is_corrupt(
    api: tuple[TestClient, NlpService, Settings],
) -> None:
    client, _, settings = api
    settings.entity_annotations_path.write_text("{ broken", encoding="utf-8")

    response = client.get("/api/nlp/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["annotation_store_readable"] is False
