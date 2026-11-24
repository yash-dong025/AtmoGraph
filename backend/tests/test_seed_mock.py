from scripts import seed_graph


def test_seed_calls_database(monkeypatch, capsys):
    calls = []
    def fake_query(cypher, params=None, *, read=True):
        calls.append((cypher, params))
        if 'RETURN entities' in cypher:
            return [{"entities":17, "relationships":16}]
        return []
    monkeypatch.setattr(seed_graph, "initialize_schema", lambda: None)
    monkeypatch.setattr(seed_graph.db, "query", fake_query)
    monkeypatch.setattr(seed_graph, "create_relation", lambda data: {"ok":True})
    seed_graph.seed()
    out=capsys.readouterr().out
    assert "SUCCESS: Seeded 17 sample nodes and 16 links." in out
    assert len(calls) == 18
