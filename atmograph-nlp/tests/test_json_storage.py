"""Tests for the atomic JSON article store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.article import Article
from app.storage.json_storage import JsonArticleStorage, StorageError


def _article(title: str, *, url: str = "", article_id: str = "") -> Article:
    return Article(article_id=article_id, title=title, url=url, source_type="mock")


def test_missing_file_is_created_automatically(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "articles.json"
    storage = JsonArticleStorage(path)

    assert storage.load() == []
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == []


def test_articles_are_added_and_reloaded(tmp_path: Path) -> None:
    storage = JsonArticleStorage(tmp_path / "articles.json")

    result = storage.add_articles([_article("First", url="https://example.com/one")])
    assert result.saved == 1
    assert result.duplicates_skipped == 0

    loaded = storage.load()
    assert len(loaded) == 1
    assert loaded[0].title == "First"
    assert storage.count() == 1


def test_existing_articles_are_preserved(tmp_path: Path) -> None:
    storage = JsonArticleStorage(tmp_path / "articles.json")
    storage.add_articles([_article("First", url="https://example.com/one")])
    storage.add_articles([_article("Second", url="https://example.com/two")])

    assert [article.title for article in storage.load()] == ["First", "Second"]


def test_url_duplicate_is_not_stored(tmp_path: Path) -> None:
    storage = JsonArticleStorage(tmp_path / "articles.json")
    storage.add_articles([_article("Original", url="https://www.example.com/story/?utm_source=rss")])

    result = storage.add_articles([_article("Same link, new headline", url="https://example.com/story")])
    assert result.saved == 0
    assert result.duplicates_skipped == 1
    assert len(storage.load()) == 1


def test_content_hash_duplicate_is_not_stored(tmp_path: Path) -> None:
    storage = JsonArticleStorage(tmp_path / "articles.json")
    storage.add_articles([_article("Port strike", url="https://example.com/a")])

    result = storage.add_articles([_article("PORT   STRIKE", url="https://example.com/b")])
    assert result.saved == 0
    assert result.duplicates_skipped == 1


def test_duplicates_within_one_batch_are_skipped(tmp_path: Path) -> None:
    storage = JsonArticleStorage(tmp_path / "articles.json")
    articles = [
        _article("Port strike", url="https://example.com/a"),
        _article("Port strike", url="https://example.com/b"),
    ]

    result = storage.add_articles(articles)
    assert result.saved == 1
    assert result.duplicates_skipped == 1


def test_invalid_json_raises_and_is_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "articles.json"
    original = "{ this is not valid json"
    path.write_text(original, encoding="utf-8")
    storage = JsonArticleStorage(path)

    with pytest.raises(StorageError):
        storage.load()
    with pytest.raises(StorageError):
        storage.add_articles([_article("New")])

    assert path.read_text(encoding="utf-8") == original


def test_non_array_json_is_rejected_safely(tmp_path: Path) -> None:
    path = tmp_path / "articles.json"
    path.write_text('{"articles": "not-a-list"}', encoding="utf-8")

    with pytest.raises(StorageError):
        JsonArticleStorage(path).load()


def test_empty_file_is_treated_as_empty_store(tmp_path: Path) -> None:
    path = tmp_path / "articles.json"
    path.write_text("", encoding="utf-8")
    storage = JsonArticleStorage(path)

    assert storage.load() == []
    assert storage.add_articles([_article("First")]).saved == 1
    assert storage.count() == 1


def test_invalid_existing_records_are_preserved_when_appending(tmp_path: Path) -> None:
    path = tmp_path / "articles.json"
    path.write_text(json.dumps([{"bad": "record"}]), encoding="utf-8")
    storage = JsonArticleStorage(path)

    assert storage.load() == []
    assert storage.add_articles([_article("Valid", url="https://example.com/valid")]).saved == 1

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw[0] == {"bad": "record"}
    assert raw[1]["title"] == "Valid"


def test_empty_batch_does_not_rewrite_the_file(tmp_path: Path) -> None:
    path = tmp_path / "articles.json"
    path.write_text("[]\n", encoding="utf-8")
    storage = JsonArticleStorage(path)

    result = storage.add_articles([])
    assert result.saved == 0
    assert result.duplicates_skipped == 0
    assert path.read_text(encoding="utf-8") == "[]\n"
