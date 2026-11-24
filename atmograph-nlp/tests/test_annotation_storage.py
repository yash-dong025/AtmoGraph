"""Tests for the separate, atomic annotation store."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.schemas.nlp import ArticleAnnotation, EntityAnnotation
from app.storage.annotation_storage import (
    AnnotationStorageError,
    JsonEntityAnnotationStorage,
)


def _entity(article_id: str, name: str = "Taiwan") -> EntityAnnotation:
    return EntityAnnotation(
        article_id=article_id,
        text=name,
        normalized_name=name,
        entity_type="COUNTRY",
        source_field="title",
        start_char=0,
        end_char=len(name),
        extraction_method="spacy",
        raw_label="GPE",
    )


def _annotation(article_id: str, entity_count: int = 1) -> ArticleAnnotation:
    entities = [_entity(article_id, name=f"Entity {index}") for index in range(entity_count)]
    return ArticleAnnotation(
        article_id=article_id,
        title=f"Article {article_id}",
        source_type="mock",
        model_name="en_core_web_sm",
        extracted_at=datetime.now(timezone.utc),
        entity_count=len(entities),
        entities=entities,
    )


def test_missing_file_is_an_empty_store(tmp_path: Path) -> None:
    path = tmp_path / "annotations.json"
    storage = JsonEntityAnnotationStorage(path)

    assert storage.load() == []
    assert storage.count() == 0
    assert storage.get("missing") is None
    assert not path.exists()


def test_upsert_get_and_count(tmp_path: Path) -> None:
    path = tmp_path / "annotations.json"
    storage = JsonEntityAnnotationStorage(path)
    annotation = _annotation("article-1", entity_count=2)

    returned = storage.upsert(annotation)

    assert returned == annotation
    assert path.exists()
    assert storage.count() == 1
    assert storage.get("article-1") == annotation
    assert storage.get("other") is None
    assert storage.load() == [annotation]


def test_upsert_replaces_existing_article(tmp_path: Path) -> None:
    path = tmp_path / "annotations.json"
    storage = JsonEntityAnnotationStorage(path)

    storage.upsert(_annotation("article-1", entity_count=1))
    storage.upsert(_annotation("article-1", entity_count=3))

    assert storage.count() == 1
    assert storage.get("article-1").entity_count == 3


def test_upsert_removes_duplicate_article_ids(tmp_path: Path) -> None:
    path = tmp_path / "annotations.json"
    storage = JsonEntityAnnotationStorage(path)
    first = _annotation("article-1", entity_count=1).model_dump(mode="json")
    second = _annotation("article-1", entity_count=2).model_dump(mode="json")
    path.write_text(json.dumps([first, second]), encoding="utf-8")

    storage.upsert(_annotation("article-1", entity_count=4))

    assert storage.count() == 1
    assert storage.get("article-1").entity_count == 4


def test_upsert_preserves_invalid_existing_records(tmp_path: Path) -> None:
    path = tmp_path / "annotations.json"
    invalid = {"unexpected": "record"}
    path.write_text(json.dumps([invalid]), encoding="utf-8")
    storage = JsonEntityAnnotationStorage(path)

    storage.upsert(_annotation("article-1"))

    assert storage.count() == 1
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert invalid in raw


def test_delete(tmp_path: Path) -> None:
    path = tmp_path / "annotations.json"
    storage = JsonEntityAnnotationStorage(path)
    storage.upsert(_annotation("article-1"))
    storage.upsert(_annotation("article-2"))

    assert storage.delete("article-1") is True
    assert storage.count() == 1
    assert storage.get("article-1") is None
    assert storage.get("article-2") is not None
    assert storage.delete("article-1") is False


def test_empty_file_is_an_empty_store(tmp_path: Path) -> None:
    path = tmp_path / "annotations.json"
    path.write_text("   \n", encoding="utf-8")
    assert JsonEntityAnnotationStorage(path).load() == []


def test_load_skips_invalid_records(tmp_path: Path) -> None:
    path = tmp_path / "annotations.json"
    valid = _annotation("article-1").model_dump(mode="json")
    path.write_text(
        json.dumps([{"article_id": "broken"}, valid]),
        encoding="utf-8",
    )

    loaded = JsonEntityAnnotationStorage(path).load()

    assert len(loaded) == 1
    assert loaded[0].article_id == "article-1"


def test_corrupt_json_raises_and_leaves_file_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "annotations.json"
    corrupt = "{ definitely not json"
    path.write_text(corrupt, encoding="utf-8")
    storage = JsonEntityAnnotationStorage(path)

    with pytest.raises(AnnotationStorageError, match="invalid JSON"):
        storage.load()
    with pytest.raises(AnnotationStorageError, match="invalid JSON"):
        storage.upsert(_annotation("article-1"))

    assert path.read_text(encoding="utf-8") == corrupt


def test_non_array_payload_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps({"unexpected": "object"}), encoding="utf-8")

    with pytest.raises(AnnotationStorageError, match="must contain a JSON array"):
        JsonEntityAnnotationStorage(path).load()


def test_atomic_write_leaves_no_temporary_files(tmp_path: Path) -> None:
    path = tmp_path / "annotations.json"
    storage = JsonEntityAnnotationStorage(path)

    storage.upsert(_annotation("article-1"))

    assert list(tmp_path.glob(".*.tmp")) == []
