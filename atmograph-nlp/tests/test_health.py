"""Phase 1 smoke tests: the API starts and reports its status."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_returns_service_info() -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["docs"] == "/docs"
    assert body["health"] == "/api/health"


def test_health_endpoint_reports_components() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["environment"] == "testing"
    assert isinstance(body["timestamp"], str)
    assert body["components"], "health response must list checked components"

    names = {component["name"] for component in body["components"]}
    assert {"fastapi", "pydantic", "spacy"}.issubset(names)

    assert set(body["features"]) == {"transformer_classifier", "graph_sync", "spacy_model"}


def test_unknown_route_returns_404() -> None:
    assert client.get("/api/does-not-exist").status_code == 404
