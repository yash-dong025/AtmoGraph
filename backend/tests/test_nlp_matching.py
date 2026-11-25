
from scripts.match_nlp_entities import match_entity


GRAPH_NODES = [
    {"id": "mumbai_port", "name": "Mumbai Port"},
    {
        "id": "bengaluru_warehouse",
        "name": "Bengaluru Warehouse"
    }
]


def test_exact_match():
    result = match_entity(
        {"normalized_name": "Mumbai Port"},
        GRAPH_NODES
    )

    assert result["status"] == "matched"
    assert result["matched_entity_id"] == "mumbai_port"


def test_case_insensitive_match():
    result = match_entity(
        {"normalized_name": "MUMBAI PORT"},
        GRAPH_NODES
    )

    assert result["status"] == "matched"


def test_unmatched_entity():
    result = match_entity(
        {"normalized_name": "Unknown Test Facility"},
        GRAPH_NODES
    )

    assert result["status"] == "unmatched"
    assert result["matched_entity_id"] is None


def test_fuzzy_match_requires_review():
    result = match_entity(
        {"normalized_name": "Mumbai Ports"},
        GRAPH_NODES
    )

    assert result["status"] == "review"
    assert result["matched_entity_id"] is None
    assert result["suggestions"][0]["id"] == "mumbai_port"
