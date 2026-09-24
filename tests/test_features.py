import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from eurostress.config import ProjectConfig
from eurostress.features import (
    MODEL_FEATURES,
    construct_weekly_features,
    sample_fridays,
    standardize_from_training,
    trailing_log_realised_volatility,
)


def _config(start: str, end: str) -> ProjectConfig:
    return ProjectConfig(
        project={
            "start_date": start,
            "end_date": end,
            "max_staleness_calendar_days": 5,
        },
        splits={
            "train_end": "2024-02-29",
            "validation_start": "2024-03-01",
            "validation_end": "2024-03-31",
            "test_start": "2024-04-01",
            "test_end": "2024-12-31",
        },
        series=(),
    )


def _frame(dates: pd.DatetimeIndex, values: np.ndarray | list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": dates, "value": values})


def _synthetic_frames(end: str = "2024-04-30") -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", end)
    sequence = np.arange(len(dates), dtype=float)
    return {
        "euro_stoxx_50": _frame(dates, 100 * np.exp(0.001 * sequence)),
        "eurusd": _frame(dates, 1.05 * np.exp(0.0002 * sequence)),
        "euro_aaa_2y": _frame(dates, 2 + 0.001 * sequence),
        "euro_aaa_10y": _frame(dates, 3 + 0.002 * sequence),
        "euro_all_ratings_10y": _frame(dates, 3.5 + 0.003 * sequence),
        "new_ciss": _frame(dates, 0.1 + 0.0001 * sequence),
    }


def test_friday_sampling_uses_only_past_and_applies_staleness_limit() -> None:
    frame = _frame(
        pd.to_datetime(["2024-01-04", "2024-01-13"]),
        [1.0, 2.0],
    )
    fridays = pd.date_range("2024-01-05", "2024-01-12", freq="W-FRI")

    result = sample_fridays(frame, fridays, max_staleness_days=5)

    assert result.loc[pd.Timestamp("2024-01-05"), "value"] == 1.0
    assert pd.isna(result.loc[pd.Timestamp("2024-01-12"), "value"])
    assert not (result["source_date"] > result.index).any()


def test_realised_volatility_uses_trailing_twenty_returns() -> None:
    dates = pd.bdate_range("2024-01-01", periods=21)
    daily_log_return = 0.01
    equity = _frame(dates, 100 * np.exp(daily_log_return * np.arange(21)))

    result = trailing_log_realised_volatility(equity)

    assert len(result) == 1
    assert np.isclose(result.iloc[0]["value"], np.log(np.sqrt(252) * daily_log_return))


def test_weekly_construction_is_unchanged_when_future_data_are_added() -> None:
    early = _synthetic_frames("2024-03-29")
    full = _synthetic_frames("2024-04-30")
    early_features, _ = construct_weekly_features(
        early, _config("2024-01-05", "2024-03-29")
    )
    full_features, _ = construct_weekly_features(
        full, _config("2024-01-05", "2024-04-26")
    )

    overlapping = full_features[
        full_features["information_date"] <= early_features["information_date"].max()
    ].reset_index(drop=True)
    assert_frame_equal(early_features, overlapping)


def test_yield_features_form_levels_before_changes() -> None:
    frames = _synthetic_frames()
    features, _ = construct_weekly_features(frames, _config("2024-01-05", "2024-04-26"))
    row = features.iloc[-1]

    # Five business-day increments: 2y=0.5 bp, slope=0.5 bp, fragmentation=0.5 bp.
    assert np.isclose(row["aaa_2y_change_bps"], 0.5)
    assert np.isclose(row["aaa_curve_slope_change_bps"], 0.5)
    assert np.isclose(row["fragmentation_spread_change_bps"], 0.5)


def test_scaling_parameters_ignore_validation_and_test_values() -> None:
    dates = pd.to_datetime(["2024-02-16", "2024-02-23", "2024-03-08", "2024-04-05"])
    weekly = pd.DataFrame({"information_date": dates, "split": ["training"] * 2 + ["validation", "locked_test"]})
    for offset, name in enumerate(MODEL_FEATURES):
        weekly[name] = [float(offset), float(offset + 2), 1_000.0, -1_000.0]

    scaled, parameters = standardize_from_training(weekly, "2024-02-29")

    for offset, name in enumerate(MODEL_FEATURES):
        assert parameters[name] == {"mean": float(offset + 1), "standard_deviation": 1.0}
        assert np.allclose(scaled.loc[:1, name], [-1.0, 1.0])
