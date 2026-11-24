from fastapi.testclient import TestClient
from app.main import app
from app.api.routes import health


def test_home():
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["module"] == "Member 1"


def test_health_with_mock_db(monkeypatch):
    monkeypatch.setattr(health.db, "verify_connection", lambda: None)
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["database"] == "connected"


def test_invalid_entity_rejected_before_db():
    with TestClient(app) as client:
        r = client.post("/api/v1/entities", json={"id":"bad space","name":"Bad", "kind":"PORT", "country":"India"})
        assert r.status_code == 422
