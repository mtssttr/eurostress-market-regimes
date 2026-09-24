"""DuckDB schema for auditable raw observations and model outputs."""

from __future__ import annotations

from pathlib import Path

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS series_catalog (
    series_id VARCHAR PRIMARY KEY,
    provider VARCHAR NOT NULL,
    description VARCHAR NOT NULL,
    unit VARCHAR NOT NULL,
    redistribution VARCHAR NOT NULL,
    source_reference VARCHAR,
    created_at_utc TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS observations (
    series_id VARCHAR NOT NULL,
    observation_date DATE NOT NULL,
    value DOUBLE NOT NULL,
    retrieved_at_utc TIMESTAMP NOT NULL,
    stale_days INTEGER,
    source_hash VARCHAR,
    PRIMARY KEY (series_id, observation_date)
);

CREATE TABLE IF NOT EXISTS features (
    information_date DATE NOT NULL,
    feature_name VARCHAR NOT NULL,
    value DOUBLE NOT NULL,
    window_start DATE,
    source_run_id VARCHAR NOT NULL,
    PRIMARY KEY (information_date, feature_name, source_run_id)
);

CREATE TABLE IF NOT EXISTS benchmark_forecasts (
    information_date DATE NOT NULL,
    target_start_exclusive DATE NOT NULL,
    target_end_inclusive DATE NOT NULL,
    target_return_count INTEGER NOT NULL,
    target_first_return_date DATE NOT NULL,
    target_last_return_date DATE NOT NULL,
    realised_variance DOUBLE NOT NULL,
    historical_variance_forecast DOUBLE NOT NULL,
    ewma_variance_forecast DOUBLE NOT NULL,
    source_run_id VARCHAR NOT NULL,
    PRIMARY KEY (information_date, source_run_id)
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id VARCHAR PRIMARY KEY,
    created_at_utc TIMESTAMP DEFAULT current_timestamp,
    train_end DATE NOT NULL,
    validation_end DATE,
    test_end DATE,
    model_name VARCHAR NOT NULL,
    configuration_json JSON NOT NULL,
    data_hash VARCHAR NOT NULL,
    code_revision VARCHAR,
    status VARCHAR NOT NULL
);
"""


def initialize_database(path: str | Path) -> Path:
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(SCHEMA)
    return database_path
