import duckdb

from eurostress.db import initialize_database


def test_database_contains_expected_tables(tmp_path) -> None:
    database = initialize_database(tmp_path / "test.duckdb")

    with duckdb.connect(str(database), read_only=True) as connection:
        names = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}

    assert names == {
        "benchmark_forecasts",
        "features",
        "model_runs",
        "observations",
        "series_catalog",
    }
