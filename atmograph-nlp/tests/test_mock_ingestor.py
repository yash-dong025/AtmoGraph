"""Tests for loading and validating the bundled mock news dataset."""

from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path

from app.config import Settings
from app.ingestion.mock_ingestor import MockIngestor

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "mock_news.json"


def _settings() -> Settings:
    return Settings(environment="testing", rss_feeds="")


def test_bundled_mock_dataset_loads() -> None:
    result = MockIngestor(settings=_settings(), path=DATASET_PATH).load()

    assert result.errors == []
    assert result.invalid_articles == 0
    assert result.total_fetched >= 15
    assert len(result.articles) == result.total_fetched
    assert all(article.source_type == "mock" for article in result.articles)
    assert len({article.article_id for article in result.articles}) == len(result.articles)
    assert all(article.url.startswith("https://example.com/atmograph/mock/") for article in result.articles)
    assert all(article.published_at and article.published_at.tzinfo == timezone.utc for article in result.articles)


def test_bundled_mock_dataset_covers_all_required_categories() -> None:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    categories = {record["category"] for record in payload}

    assert {
        "natural_disaster",
        "factory_shutdown",
        "labor_strike",
        "shipping_delay",
        "port_congestion",
        "trade_restriction",
        "raw_material_shortage",
        "transportation_disruption",
    }.issubset(categories)


def test_mock_records_can_omit_optional_fields() -> None:
    result = MockIngestor(settings=_settings(), path=DATASET_PATH).load()
    optional_missing = [article for article in result.articles if article.author is None]
    assert optional_missing, "dataset should exercise records without an author"


def test_invalid_records_are_skipped_and_counted(tmp_path: Path) -> None:
    dataset = tmp_path / "mock.json"
    dataset.write_text(
        json.dumps(
            [
                {"title": "Valid headline", "source": "Demo", "source_type": "rss"},
                {"description": "Missing title"},
                "not-an-object",
            ]
        ),
        encoding="utf-8",
    )

    result = MockIngestor(settings=_settings(), path=dataset).load()

    assert result.total_fetched == 3
    assert result.invalid_articles == 2
    assert len(result.articles) == 1
    assert result.articles[0].source_type == "mock", "mock loader must force source_type"
    assert result.errors


def test_missing_dataset_is_reported_without_raising(tmp_path: Path) -> None:
    result = MockIngestor(settings=_settings(), path=tmp_path / "missing.json").load()

    assert result.articles == []
    assert result.total_fetched == 0
    assert result.errors


def test_invalid_json_is_reported_without_raising(tmp_path: Path) -> None:
    dataset = tmp_path / "broken.json"
    dataset.write_text("{not valid json", encoding="utf-8")

    result = MockIngestor(settings=_settings(), path=dataset).load()

    assert result.articles == []
    assert result.errors
