"""Unit tests for entity-name cleaning and alias resolution."""

from __future__ import annotations

from app.nlp.normalizer import canonical_key, normalize_entity_name


def test_display_name_strips_articles_and_whitespace() -> None:
    assert normalize_entity_name("  The   World Trade Organization ") == "World Trade Organization"
    assert normalize_entity_name("the red sea") == "Red Sea"
    assert normalize_entity_name("  Port of   Shanghai ") == "Port of Shanghai"


def test_country_abbreviations_are_resolved() -> None:
    assert normalize_entity_name("U.S.") == "United States"
    assert normalize_entity_name("U.S.A.") == "United States"
    assert normalize_entity_name("USA") == "United States"
    assert normalize_entity_name("UK") == "United Kingdom"
    assert normalize_entity_name("UAE") == "United Arab Emirates"
    assert normalize_entity_name("Korea") == "South Korea"


def test_material_spelling_variants_are_resolved() -> None:
    assert normalize_entity_name("Aluminium") == "Aluminum"
    assert normalize_entity_name("aluminum") == "Aluminum"
    assert normalize_entity_name("rare earths") == "Rare Earth"


def test_organization_acronyms_are_resolved() -> None:
    assert normalize_entity_name("WTO") == "World Trade Organization"
    assert normalize_entity_name("WHO") == "World Health Organization"
    assert normalize_entity_name("EU") == "European Union"
    assert normalize_entity_name("UN") == "United Nations"


def test_canonical_key_matches_aliases() -> None:
    assert canonical_key("U.S.") == canonical_key("United States") == "united states"
    assert canonical_key("Aluminium") == canonical_key("Aluminum") == "aluminum"
    assert canonical_key("WTO") == canonical_key("World Trade Organization")
    assert canonical_key("China's") == canonical_key("China") == "china"


def test_unknown_names_keep_their_surface_spelling() -> None:
    assert normalize_entity_name("ACME Widgets") == "ACME Widgets"
    assert normalize_entity_name("Mei-Ling Chen") == "Mei-Ling Chen"


def test_empty_and_punctuation_only_names_become_empty() -> None:
    assert normalize_entity_name("") == ""
    assert normalize_entity_name("   ") == ""
    assert normalize_entity_name("...") == ""
