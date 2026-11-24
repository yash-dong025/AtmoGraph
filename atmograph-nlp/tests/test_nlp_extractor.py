"""Unit tests for spaCy NER + supply-chain rule extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.nlp.extractor import ArticleEntityExtractor
from app.nlp.patterns import PatternConfigError, build_patterns
from app.nlp.taxonomy import ENTITY_TYPES
from app.schemas.article import Article


def _settings(**overrides) -> Settings:
    values = {"environment": "testing"}
    values.update(overrides)
    return Settings(**values)


def _article(title: str, description: str = "") -> Article:
    return Article(
        article_id="test-article",
        title=title,
        description=description,
        source="Unit Test",
        source_type="mock",
    )


@pytest.fixture(scope="module")
def extractor() -> ArticleEntityExtractor:
    return ArticleEntityExtractor(_settings())


def test_entities_have_types_offsets_and_provenance(extractor: ArticleEntityExtractor) -> None:
    article = _article(
        title="Port of Shanghai congestion hits semiconductor exports",
        description=(
            "TSMC said a fire at a plant in Taiwan delayed shipments of steel and automotive parts. "
            "The World Trade Organization met in Geneva on 24 November 2026. "
            "U.S. officials and China discussed tariffs. Foxconn halted production in Zhengzhou. "
            "Aluminium and cobalt supplies tightened for battery manufacturers. "
            "The Port of Los Angeles and the container terminal were congested."
        ),
    )

    entities = extractor.extract(article)
    pairs = {(entity.entity_type, entity.normalized_name) for entity in entities}

    expected = {
        ("PORT", "Port of Shanghai"),
        ("INDUSTRY", "Semiconductor"),
        ("COMPANY", "TSMC"),
        ("FACILITY", "Plant"),
        ("COUNTRY", "Taiwan"),
        ("MATERIAL", "Steel"),
        ("INDUSTRY", "Automotive"),
        ("ORGANIZATION", "World Trade Organization"),
        ("LOCATION", "Geneva"),
        ("DATE", "24 November 2026"),
        ("COUNTRY", "United States"),
        ("COUNTRY", "China"),
        ("COMPANY", "Foxconn"),
        ("LOCATION", "Zhengzhou"),
        ("MATERIAL", "Cobalt"),
        ("PRODUCT", "Battery"),
        ("PORT", "Port of Los Angeles"),
        ("FACILITY", "Container Terminal"),
    }
    assert expected.issubset(pairs)

    for entity in entities:
        assert entity.article_id == article.article_id
        assert entity.entity_type in ENTITY_TYPES
        assert entity.normalized_name
        assert entity.extraction_method in {"spacy", "rule"}
        assert entity.end_char > entity.start_char
        if entity.source_field == "title":
            assert entity.text == article.title[entity.start_char : entity.end_char]
        else:
            assert entity.text == article.description[entity.start_char : entity.end_char]


def test_mentions_are_unique_and_sorted(extractor: ArticleEntityExtractor) -> None:
    article = _article(
        title="Steel and aluminum supplies tighten",
        description="Steel buyers in China and the U.S. face delays.",
    )
    entities = extractor.extract(article)

    signatures = [
        (entity.entity_type, entity.normalized_name, entity.source_field, entity.start_char, entity.end_char)
        for entity in entities
    ]
    assert len(signatures) == len(set(signatures))

    title_entities = [entity for entity in entities if entity.source_field == "title"]
    assert title_entities == sorted(title_entities, key=lambda entity: (entity.start_char, entity.end_char))
    assert entities[: len(title_entities)] == title_entities


def test_country_and_organization_aliases_are_normalized(extractor: ArticleEntityExtractor) -> None:
    article = _article(
        title="Tariff talks",
        description="The U.S. and China discussed tariffs with the WTO.",
    )
    entities = extractor.extract(article)
    by_text = {entity.text: entity for entity in entities}

    assert by_text["U.S."].entity_type == "COUNTRY"
    assert by_text["U.S."].normalized_name == "United States"
    assert by_text["China"].entity_type == "COUNTRY"
    assert by_text["WTO"].entity_type == "ORGANIZATION"
    assert by_text["WTO"].normalized_name == "World Trade Organization"


def test_supply_chain_rules_detect_ports_facilities_and_materials(
    extractor: ArticleEntityExtractor,
) -> None:
    article = _article(
        title="Port of Rotterdam and Kaohsiung Terminal handle steel and automotive cargo"
    )
    pairs = {(entity.entity_type, entity.normalized_name) for entity in extractor.extract(article)}

    assert ("PORT", "Port of Rotterdam") in pairs
    assert ("FACILITY", "Kaohsiung Terminal") in pairs
    assert ("MATERIAL", "Steel") in pairs
    assert ("INDUSTRY", "Automotive") in pairs


def test_headline_fragments_are_not_reported_as_organizations(
    extractor: ArticleEntityExtractor,
) -> None:
    article = _article(title="Fire Forces Temporary Closure of Auto Parts Stamping Plant")
    entities = extractor.extract(article)

    name_entities = [
        entity for entity in entities if entity.entity_type in {"ORGANIZATION", "COMPANY", "PERSON"}
    ]
    assert all("Fire Forces" not in entity.text for entity in name_entities)
    assert any(
        entity.entity_type == "FACILITY" and "Stamping Plant" in entity.normalized_name
        for entity in entities
    )


def test_typhoon_headline_is_not_reported_as_an_organization(
    extractor: ArticleEntityExtractor,
) -> None:
    article = _article(title="Typhoon Halts Container Operations at Kaohsiung Terminal")
    entities = extractor.extract(article)

    assert not any(
        entity.entity_type in {"ORGANIZATION", "COMPANY", "PERSON"}
        and "Typhoon Halts" in entity.text
        for entity in entities
    )
    assert any(
        entity.entity_type == "FACILITY" and entity.normalized_name == "Kaohsiung Terminal"
        for entity in entities
    )


def test_person_names_are_kept(extractor: ArticleEntityExtractor) -> None:
    article = _article(
        title="Plant update",
        description="Mei-Ling Chen said the plant would reopen next week.",
    )
    entities = extractor.extract(article)
    assert any(
        entity.entity_type == "PERSON" and entity.normalized_name == "Mei-Ling Chen"
        for entity in entities
    )


def test_neutral_text_has_no_entities(extractor: ArticleEntityExtractor) -> None:
    article = _article(title="The report was published.", description="Nothing changed.")
    assert extractor.extract(article) == []


def test_extraction_is_deterministic(extractor: ArticleEntityExtractor) -> None:
    article = _article(
        title="Port of Singapore congestion",
        description="Steel and semiconductor shipments were delayed in Taiwan.",
    )
    first = [
        (entity.entity_type, entity.text, entity.start_char, entity.end_char)
        for entity in extractor.extract(article)
    ]
    second = [
        (entity.entity_type, entity.text, entity.start_char, entity.end_char)
        for entity in extractor.extract(article)
    ]
    assert first == second


def test_custom_pattern_file_adds_domain_terms(tmp_path: Path) -> None:
    pattern_file = tmp_path / "patterns.json"
    pattern_file.write_text(
        json.dumps(
            {
                "patterns": [
                    {"label": "MATERIAL", "pattern": "unobtainium"},
                    {
                        "label": "FACILITY",
                        "pattern": [{"LOWER": "orbital"}, {"LOWER": "depot"}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    extractor = ArticleEntityExtractor(_settings(nlp_patterns_path=pattern_file))
    article = _article(
        title="Unobtainium shortage",
        description="A shortage of unobtainium hit the orbital depot.",
    )
    pairs = {(entity.entity_type, entity.text.lower()) for entity in extractor.extract(article)}

    assert ("MATERIAL", "unobtainium") in pairs
    assert ("FACILITY", "orbital depot") in pairs


def test_invalid_custom_pattern_label_is_rejected(tmp_path: Path) -> None:
    pattern_file = tmp_path / "patterns.json"
    pattern_file.write_text(
        json.dumps([{"label": "NOT_A_TAXONOMY_TYPE", "pattern": "example"}]),
        encoding="utf-8",
    )
    with pytest.raises(PatternConfigError):
        build_patterns(pattern_file)
