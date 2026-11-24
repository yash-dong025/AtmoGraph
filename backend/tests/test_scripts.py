from pathlib import Path
from scripts.seed_graph import DATA_DIR


def test_seed_files_exist_and_contain_rows():
    assert (DATA_DIR / "nodes.csv").exists()
    assert (DATA_DIR / "relationships.csv").exists()
    assert len((DATA_DIR / "nodes.csv").read_text().splitlines()) > 10
    assert len((DATA_DIR / "relationships.csv").read_text().splitlines()) > 10


def test_schema_statements_are_defined():
    from app.database.schema import CONSTRAINTS, INDEXES
    assert len(CONSTRAINTS) >= 3
    assert len(INDEXES) >= 2
