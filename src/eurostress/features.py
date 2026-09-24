"""Point-in-time weekly feature construction."""

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

FEATURE_VERSION = "1.0.0"
MODEL_FEATURES = (
    "equity_return_1w",
    "equity_log_rv_20d",
    "eurusd_return_1w",
    "aaa_2y_change_bps",
    "aaa_curve_slope_change_bps",
    "fragmentation_spread_change_bps",
)


@dataclass(frozen=True)
class FeatureBuildResult:
    run_id: str
    first_information_date: str
    last_information_date: str
    rows: int
    training_rows: int
    validation_rows: int
    locked_test_rows: int


def _prepare_observations(frame: pd.DataFrame, series_name: str) -> pd.DataFrame:
    required = {"date", "value"}
    if not required <= set(frame.columns):
        raise ValueError(f"{series_name} is missing columns: {sorted(required - set(frame.columns))}")

    observations = frame.loc[:, ["date", "value"]].copy()
    observations["date"] = pd.to_datetime(observations["date"], errors="raise").dt.tz_localize(
        None
    )
    observations["value"] = pd.to_numeric(observations["value"], errors="raise")
    observations = observations.sort_values("date")
    if observations["date"].duplicated().any():
        raise ValueError(f"{series_name} contains duplicate dates")
    if observations["value"].isna().any() or observations.empty:
        raise ValueError(f"{series_name} contains missing values or no observations")
    return observations


def sample_fridays(
    frame: pd.DataFrame,
    fridays: pd.DatetimeIndex,
    max_staleness_days: int,
) -> pd.DataFrame:
    """Select the final observation available on or before each Friday."""
    indexed = frame.set_index("date").sort_index()
    indexed["source_date"] = indexed.index
    weekly = indexed.loc[:, ["value", "source_date"]].resample("W-FRI").last()
    weekly = weekly.reindex(fridays)
    weekly.index.name = "information_date"
    weekly["stale_days"] = (weekly.index.to_series() - weekly["source_date"]).dt.days
    invalid = weekly["stale_days"] > max_staleness_days
    weekly.loc[invalid, ["value", "source_date", "stale_days"]] = np.nan
    return weekly


def trailing_log_realised_volatility(
    equity: pd.DataFrame, window: int = 20, annualisation_days: int = 252
) -> pd.DataFrame:
    """Compute log annualised RMS volatility from trailing daily log returns."""
    result = equity.copy()
    returns = np.log(result["value"] / result["value"].shift(1))
    variance = returns.pow(2).rolling(window=window, min_periods=window).mean()
    result["value"] = np.log(np.sqrt(annualisation_days * variance))
    return result.dropna(subset=["value"])


def assign_split(dates: pd.Series, config: ProjectConfig) -> pd.Series:
    values = pd.to_datetime(dates)
    train_end = pd.Timestamp(config.splits["train_end"])
    validation_start = pd.Timestamp(config.splits["validation_start"])
    validation_end = pd.Timestamp(config.splits["validation_end"])
    test_start = pd.Timestamp(config.splits["test_start"])
    test_end = pd.Timestamp(config.splits["test_end"])

    labels = np.select(
        [
            values <= train_end,
            (values >= validation_start) & (values <= validation_end),
            (values >= test_start) & (values <= test_end),
        ],
        ["training", "validation", "locked_test"],
        default="outside_frozen_splits",
    )
    return pd.Series(labels, index=dates.index, dtype="string")


def construct_weekly_features(
    raw_frames: dict[str, pd.DataFrame], config: ProjectConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct the six frozen model inputs and the external CISS series."""
    required_series = {
        "euro_stoxx_50",
        "eurusd",
        "euro_aaa_2y",
        "euro_aaa_10y",
        "euro_all_ratings_10y",
        "new_ciss",
    }
    missing = required_series - raw_frames.keys()
    if missing:
        raise ValueError(f"Missing raw series: {sorted(missing)}")

    prepared = {
        name: _prepare_observations(raw_frames[name], name) for name in required_series
    }
    fridays = pd.date_range(
        start=config.project["start_date"], end=config.project["end_date"], freq="W-FRI"
    )
    max_staleness = int(config.project["max_staleness_calendar_days"])
    sampled = {
        name: sample_fridays(frame, fridays, max_staleness)
        for name, frame in prepared.items()
    }
    equity_volatility = sample_fridays(
        trailing_log_realised_volatility(prepared["euro_stoxx_50"]),
        fridays,
        max_staleness,
    )

    weekly = pd.DataFrame(index=fridays)
    weekly.index.name = "information_date"
    weekly["equity_return_1w"] = np.log(
        sampled["euro_stoxx_50"]["value"]
        / sampled["euro_stoxx_50"]["value"].shift(1)
    )
    weekly["equity_log_rv_20d"] = equity_volatility["value"]
    weekly["eurusd_return_1w"] = np.log(
        sampled["eurusd"]["value"] / sampled["eurusd"]["value"].shift(1)
    )
    weekly["aaa_2y_change_bps"] = 100 * sampled["euro_aaa_2y"]["value"].diff()

    aaa_slope = (
        sampled["euro_aaa_10y"]["value"] - sampled["euro_aaa_2y"]["value"]
    )
    weekly["aaa_curve_slope_change_bps"] = 100 * aaa_slope.diff()

    fragmentation_spread = (
        sampled["euro_all_ratings_10y"]["value"]
        - sampled["euro_aaa_10y"]["value"]
    )
    weekly["fragmentation_spread_change_bps"] = 100 * fragmentation_spread.diff()
    weekly["new_ciss_external"] = sampled["new_ciss"]["value"]

    source_dates = pd.DataFrame(index=fridays)
    source_dates.index.name = "information_date"
    for name in sorted(required_series):
        source_dates[f"{name}_source_date"] = sampled[name]["source_date"]
        source_dates[f"{name}_stale_days"] = sampled[name]["stale_days"].astype("Int64")

    weekly = weekly.dropna(subset=[*MODEL_FEATURES, "new_ciss_external"]).reset_index()
    weekly["split"] = assign_split(weekly["information_date"], config)
    if (weekly["split"] == "outside_frozen_splits").any():
        raise ValueError("Feature rows fall outside the frozen chronological splits")
    if not np.isfinite(weekly.loc[:, MODEL_FEATURES].to_numpy()).all():
        raise ValueError("Model features contain non-finite values")

    source_dates = source_dates.loc[weekly["information_date"]].reset_index()
    if (source_dates.filter(like="_source_date").gt(source_dates["information_date"], axis=0)).any(
        axis=None
    ):
        raise ValueError("A weekly feature uses an observation from after its information date")
    return weekly, source_dates


def standardize_from_training(
    weekly: pd.DataFrame, train_end: str
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Standardize model inputs using parameters estimated only through train_end."""
    training = weekly[weekly["information_date"] <= pd.Timestamp(train_end)]
    if training.empty:
        raise ValueError("No training rows are available for scaling")

    means = training.loc[:, MODEL_FEATURES].mean()
    standard_deviations = training.loc[:, MODEL_FEATURES].std(ddof=0)
    if (standard_deviations <= 0).any():
        raise ValueError("At least one training feature has zero variance")

    scaled = weekly.loc[:, ["information_date", "split"]].copy()
    scaled.loc[:, MODEL_FEATURES] = (
        weekly.loc[:, MODEL_FEATURES] - means
    ) / standard_deviations
    parameters = {
        name: {"mean": float(means[name]), "standard_deviation": float(standard_deviations[name])}
        for name in MODEL_FEATURES
    }
    return scaled, parameters


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_id(config_path: Path, manifest_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(FEATURE_VERSION.encode())
    digest.update(config_path.read_bytes())
    digest.update(manifest_path.read_bytes())
    return f"features-{digest.hexdigest()[:16]}"


def build_feature_dataset(
    config: ProjectConfig,
    config_path: str | Path,
    input_directory: str | Path,
    output_directory: str | Path,
    database_path: str | Path,
) -> FeatureBuildResult:
    input_path = Path(input_directory)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = input_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing raw-data manifest: {manifest_path}")

    raw_frames = {
        series.name: pd.read_parquet(input_path / f"{series.name}.parquet")
        for series in config.series
    }
    weekly, provenance = construct_weekly_features(raw_frames, config)
    scaled, scaling_parameters = standardize_from_training(weekly, config.splits["train_end"])
    run_id = _run_id(Path(config_path), manifest_path)

    weekly_path = output_path / "weekly_features.parquet"
    scaled_path = output_path / "weekly_features_scaled.parquet"
    provenance_path = output_path / "weekly_source_provenance.parquet"
    weekly.to_parquet(weekly_path, index=False)
    scaled.to_parquet(scaled_path, index=False)
    provenance.to_parquet(provenance_path, index=False)

    manifest = {
        "feature_version": FEATURE_VERSION,
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input_manifest_sha256": _sha256(manifest_path),
        "configuration_sha256": _sha256(Path(config_path)),
        "model_features": list(MODEL_FEATURES),
        "external_validation_columns": ["new_ciss_external"],
        "scaling_fit_end": config.splits["train_end"],
        "scaling_parameters": scaling_parameters,
        "outputs": {
            weekly_path.name: _sha256(weekly_path),
            scaled_path.name: _sha256(scaled_path),
            provenance_path.name: _sha256(provenance_path),
        },
    }
    (output_path / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    database = initialize_database(database_path)
    long_features = weekly.melt(
        id_vars=["information_date"],
        value_vars=list(MODEL_FEATURES),
        var_name="feature_name",
        value_name="value",
    )
    long_features["window_start"] = pd.NaT
    long_features["source_run_id"] = run_id
    with duckdb.connect(str(database)) as connection:
        connection.execute("DELETE FROM features WHERE source_run_id = ?", [run_id])
        connection.register("feature_rows", long_features)
        connection.execute(
            """
            INSERT INTO features
            SELECT information_date, feature_name, value, window_start, source_run_id
            FROM feature_rows
            """
        )

    counts = weekly["split"].value_counts()
    result = FeatureBuildResult(
        run_id=run_id,
        first_information_date=weekly["information_date"].min().date().isoformat(),
        last_information_date=weekly["information_date"].max().date().isoformat(),
        rows=len(weekly),
        training_rows=int(counts.get("training", 0)),
        validation_rows=int(counts.get("validation", 0)),
        locked_test_rows=int(counts.get("locked_test", 0)),
    )
    (output_path / "build_summary.json").write_text(
        json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8"
    )
    return result
