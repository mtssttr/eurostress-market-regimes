"""Point-in-time realised-variance targets and benchmark forecasts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from eurostress.config import ProjectConfig
from eurostress.db import initialize_database

BENCHMARK_VERSION = "1.0.0"


@dataclass(frozen=True)
class BenchmarkBuildResult:
    run_id: str
    first_information_date: str
    last_information_date: str
    rows: int
    training_rows: int
    validation_rows: int
    locked_test_rows: int


def daily_log_returns(equity: pd.DataFrame) -> pd.DataFrame:
    """Return one log return for each observed trading date after the first."""
    required = {"date", "value"}
    if not required <= set(equity.columns):
        raise ValueError(f"Equity data are missing columns: {sorted(required - set(equity.columns))}")

    prices = equity.loc[:, ["date", "value"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="raise").dt.tz_localize(None)
    prices["value"] = pd.to_numeric(prices["value"], errors="raise")
    prices = prices.sort_values("date")
    if prices["date"].duplicated().any():
        raise ValueError("Equity data contain duplicate dates")
    if prices.empty or prices["value"].isna().any() or (prices["value"] <= 0).any():
        raise ValueError("Equity prices must be non-empty, positive, and finite")

    returns = prices.loc[:, ["date"]].copy()
    returns["log_return"] = np.log(prices["value"] / prices["value"].shift(1))
    returns = returns.dropna().reset_index(drop=True)
    if not np.isfinite(returns["log_return"]).all():
        raise ValueError("Daily equity returns contain non-finite values")
    return returns


def historical_variance_series(
    returns: pd.DataFrame, window: int, annualization_days: int
) -> pd.DataFrame:
    """Annualized trailing mean squared return known on each trading date."""
    result = returns.loc[:, ["date"]].copy()
    result["value"] = (
        annualization_days
        * returns["log_return"].pow(2).rolling(window=window, min_periods=window).mean()
    )
    return result.dropna().reset_index(drop=True)


def ewma_variance_series(
    returns: pd.DataFrame,
    decay: float,
    initialization_window: int,
    annualization_days: int,
) -> pd.DataFrame:
    """Annualized filtered EWMA variance known after each daily return."""
    if not 0 < decay < 1:
        raise ValueError("EWMA decay must lie strictly between zero and one")
    if len(returns) < initialization_window:
        raise ValueError("Insufficient returns for EWMA initialization")

    squared = returns["log_return"].pow(2).to_numpy()
    values = np.full(len(returns), np.nan)
    initial_index = initialization_window - 1
    values[initial_index] = squared[:initialization_window].mean()
    for index in range(initial_index + 1, len(squared)):
        values[index] = decay * values[index - 1] + (1 - decay) * squared[index]

    result = returns.loc[:, ["date"]].copy()
    result["value"] = annualization_days * values
    return result.dropna().reset_index(drop=True)


def _asof_values(series: pd.DataFrame, information_dates: pd.Series) -> pd.DataFrame:
    """Select the final dated value available on or before each information date."""
    left = pd.DataFrame({"information_date": pd.to_datetime(information_dates)}).sort_values(
        "information_date"
    )
    right = series.rename(columns={"date": "source_date"}).sort_values("source_date")
    return pd.merge_asof(
        left,
        right,
        left_on="information_date",
        right_on="source_date",
        direction="backward",
        allow_exact_matches=True,
    )


def next_week_realised_variance(
    returns: pd.DataFrame,
    information_dates: pd.Series,
    horizon_calendar_days: int,
    annualization_days: int,
    sample_end: str,
    minimum_return_count: int = 1,
) -> pd.DataFrame:
    """Build annualized targets strictly after each Friday through the next Friday."""
    rows: list[dict[str, object]] = []
    final_date = pd.Timestamp(sample_end)
    dates = returns["date"]
    squared = returns["log_return"].pow(2)

    for raw_information_date in pd.to_datetime(information_dates):
        information_date = pd.Timestamp(raw_information_date)
        target_end = information_date + pd.to_timedelta(horizon_calendar_days, unit="D")
        if target_end > final_date:
            continue
        in_window = (dates > information_date) & (dates <= target_end)
        count = int(in_window.sum())
        if count < minimum_return_count:
            continue
        rows.append(
            {
                "information_date": information_date,
                "target_start_exclusive": information_date,
                "target_end_inclusive": target_end,
                "target_return_count": count,
                "target_first_return_date": dates.loc[in_window].min(),
                "target_last_return_date": dates.loc[in_window].max(),
                "realised_variance": annualization_days * float(squared.loc[in_window].mean()),
            }
        )
    return pd.DataFrame(rows)


def construct_targets_and_benchmarks(
    equity: pd.DataFrame,
    weekly_features: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Align next-week targets with forecasts made at each Friday cutoff."""
    settings = config.evaluation
    required_settings = {
        "target_horizon_calendar_days",
        "minimum_target_returns",
        "annualization_trading_days",
        "historical_variance_window_trading_days",
        "ewma_lambda",
        "ewma_initialization_trading_days",
    }
    if not required_settings <= settings.keys():
        raise ValueError(f"Missing evaluation settings: {sorted(required_settings - settings.keys())}")
    if not {"information_date", "split"} <= set(weekly_features.columns):
        raise ValueError("Weekly features require information_date and split columns")

    returns = daily_log_returns(equity)
    annualization = int(settings["annualization_trading_days"])
    historical = historical_variance_series(
        returns,
        window=int(settings["historical_variance_window_trading_days"]),
        annualization_days=annualization,
    )
    ewma = ewma_variance_series(
        returns,
        decay=float(settings["ewma_lambda"]),
        initialization_window=int(settings["ewma_initialization_trading_days"]),
        annualization_days=annualization,
    )
    information_dates = weekly_features["information_date"]
    targets = next_week_realised_variance(
        returns,
        information_dates,
        horizon_calendar_days=int(settings["target_horizon_calendar_days"]),
        annualization_days=annualization,
        sample_end=str(config.project["end_date"]),
        minimum_return_count=int(settings["minimum_target_returns"]),
    )
    historical_weekly = _asof_values(historical, information_dates).rename(
        columns={"source_date": "historical_source_date", "value": "historical_variance_forecast"}
    )
    ewma_weekly = _asof_values(ewma, information_dates).rename(
        columns={"source_date": "ewma_source_date", "value": "ewma_variance_forecast"}
    )

    result = weekly_features.loc[:, ["information_date", "split"]].merge(
        targets, on="information_date", how="inner", validate="one_to_one"
    )
    result = result.merge(historical_weekly, on="information_date", validate="one_to_one")
    result = result.merge(ewma_weekly, on="information_date", validate="one_to_one")
    result = result.sort_values("information_date").reset_index(drop=True)

    value_columns = [
        "realised_variance",
        "historical_variance_forecast",
        "ewma_variance_forecast",
    ]
    result = result.dropna(subset=value_columns)
    if not np.isfinite(result[value_columns].to_numpy()).all():
        raise ValueError("Targets or forecasts contain non-finite values")
    if (result[value_columns] <= 0).any(axis=None):
        raise ValueError("Targets and variance forecasts must be positive")
    if (result["historical_source_date"] > result["information_date"]).any():
        raise ValueError("Historical forecast uses future data")
    if (result["ewma_source_date"] > result["information_date"]).any():
        raise ValueError("EWMA forecast uses future data")
    if not (result["target_start_exclusive"] == result["information_date"]).all():
        raise ValueError("Target window is not aligned to the information date")
    if not (result["target_end_inclusive"] > result["information_date"]).all():
        raise ValueError("Target does not lie after the forecast date")
    if not (result["target_first_return_date"] > result["information_date"]).all():
        raise ValueError("Target includes a return known at forecast time")
    if not (result["target_last_return_date"] <= result["target_end_inclusive"]).all():
        raise ValueError("Target includes a return after the target window")
    max_staleness = int(config.project.get("max_staleness_calendar_days", 5))
    if ((result["information_date"] - result["historical_source_date"]).dt.days > max_staleness).any():
        raise ValueError("Historical forecast is stale")
    if ((result["information_date"] - result["ewma_source_date"]).dt.days > max_staleness).any():
        raise ValueError("EWMA forecast is stale")
    if ((result["target_end_inclusive"] - result["target_last_return_date"]).dt.days > max_staleness).any():
        raise ValueError("Target window is incomplete or stale")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_id(config_path: Path, raw_manifest: Path, feature_manifest: Path) -> str:
    feature_metadata = json.loads(feature_manifest.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    digest.update(BENCHMARK_VERSION.encode())
    digest.update(config_path.read_bytes())
    digest.update(raw_manifest.read_bytes())
    digest.update(feature_metadata["run_id"].encode())
    digest.update(feature_metadata["outputs"]["weekly_features.parquet"].encode())
    return f"benchmarks-{digest.hexdigest()[:16]}"


def build_benchmark_dataset(
    config: ProjectConfig,
    config_path: str | Path,
    raw_directory: str | Path,
    processed_directory: str | Path,
    database_path: str | Path,
) -> BenchmarkBuildResult:
    raw_path = Path(raw_directory)
    processed_path = Path(processed_directory)
    raw_manifest = raw_path / "manifest.json"
    feature_manifest = processed_path / "feature_manifest.json"
    feature_path = processed_path / "weekly_features.parquet"
    for required in (raw_manifest, feature_manifest, feature_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing Phase 3 input: {required}")

    equity = pd.read_parquet(raw_path / "euro_stoxx_50.parquet")
    weekly_features = pd.read_parquet(feature_path)
    dataset = construct_targets_and_benchmarks(equity, weekly_features, config)
    run_id = _run_id(Path(config_path), raw_manifest, feature_manifest)

    output_path = processed_path / "weekly_targets_benchmarks.parquet"
    dataset.to_parquet(output_path, index=False)
    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "configuration_sha256": _sha256(Path(config_path)),
        "raw_manifest_sha256": _sha256(raw_manifest),
        "feature_manifest_sha256": _sha256(feature_manifest),
        "parameters": dict(config.evaluation),
        "output": {output_path.name: _sha256(output_path)},
        "performance_metrics_computed": False,
    }
    (processed_path / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    database = initialize_database(database_path)
    database_rows = dataset.loc[
        :,
        [
            "information_date",
            "target_start_exclusive",
            "target_end_inclusive",
            "target_return_count",
            "target_first_return_date",
            "target_last_return_date",
            "realised_variance",
            "historical_variance_forecast",
            "ewma_variance_forecast",
        ],
    ].copy()
    database_rows["source_run_id"] = run_id
    with duckdb.connect(str(database)) as connection:
        connection.execute("DELETE FROM benchmark_forecasts WHERE source_run_id = ?", [run_id])
        connection.register("benchmark_rows", database_rows)
        connection.execute(
            """
            INSERT INTO benchmark_forecasts
            SELECT information_date, target_start_exclusive, target_end_inclusive,
                   target_return_count, target_first_return_date, target_last_return_date,
                   realised_variance, historical_variance_forecast, ewma_variance_forecast,
                   source_run_id
            FROM benchmark_rows
            """
        )

    counts = dataset["split"].value_counts()
    result = BenchmarkBuildResult(
        run_id=run_id,
        first_information_date=dataset["information_date"].min().date().isoformat(),
        last_information_date=dataset["information_date"].max().date().isoformat(),
        rows=len(dataset),
        training_rows=int(counts.get("training", 0)),
        validation_rows=int(counts.get("validation", 0)),
        locked_test_rows=int(counts.get("locked_test", 0)),
    )
    (processed_path / "benchmark_build_summary.json").write_text(
        json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8"
    )
    return result
