import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from eurostress.benchmarks import (
    construct_targets_and_benchmarks,
    daily_log_returns,
    ewma_variance_series,
    historical_variance_series,
    next_week_realised_variance,
)
from eurostress.config import ProjectConfig


def _config(end: str = "2024-06-30") -> ProjectConfig:
    return ProjectConfig(
        project={"end_date": end},
        splits={},
        series=(),
        evaluation={
            "target_horizon_calendar_days": 7,
            "minimum_target_returns": 2,
            "annualization_trading_days": 252,
            "historical_variance_window_trading_days": 20,
            "ewma_lambda": 0.94,
            "ewma_initialization_trading_days": 20,
        },
    )


def _equity(end: str = "2024-06-30") -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", end)
    increments = 0.001 + 0.00001 * np.arange(len(dates))
    prices = 100 * np.exp(np.cumsum(increments))
    return pd.DataFrame({"date": dates, "value": prices})


def _weekly_features(end: str = "2024-06-21") -> pd.DataFrame:
    dates = pd.date_range("2024-02-02", end, freq="W-FRI")
    return pd.DataFrame({"information_date": dates, "split": "training"})


def test_next_week_target_uses_returns_strictly_after_information_date() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-12"]),
            "log_return": [0.5, 0.01, -0.02],
        }
    )

    target = next_week_realised_variance(
        returns,
        pd.Series(pd.to_datetime(["2024-01-05"])),
        horizon_calendar_days=7,
        annualization_days=252,
        sample_end="2024-01-31",
    ).iloc[0]

    assert target["target_return_count"] == 2
    assert np.isclose(target["realised_variance"], 252 * (0.01**2 + 0.02**2) / 2)
    assert target["target_start_exclusive"] == pd.Timestamp("2024-01-05")
    assert target["target_end_inclusive"] == pd.Timestamp("2024-01-12")
    assert target["target_first_return_date"] == pd.Timestamp("2024-01-08")
    assert target["target_last_return_date"] == pd.Timestamp("2024-01-12")


def test_historical_variance_uses_exactly_trailing_returns() -> None:
    dates = pd.bdate_range("2024-01-01", periods=4)
    returns = pd.DataFrame({"date": dates, "log_return": [0.01, 0.02, 0.03, 0.04]})

    result = historical_variance_series(returns, window=3, annualization_days=252)

    assert len(result) == 2
    assert np.isclose(result.iloc[0]["value"], 252 * np.mean(np.square([0.01, 0.02, 0.03])))
    assert np.isclose(result.iloc[1]["value"], 252 * np.mean(np.square([0.02, 0.03, 0.04])))


def test_target_rejects_a_week_with_fewer_than_two_returns() -> None:
    returns = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-05", "2024-01-08"]), "log_return": [0.1, 0.01]}
    )

    target = next_week_realised_variance(
        returns,
        pd.Series(pd.to_datetime(["2024-01-05"])),
        horizon_calendar_days=7,
        annualization_days=252,
        sample_end="2024-01-31",
        minimum_return_count=2,
    )

    assert target.empty


def test_ewma_initialization_and_recursion_are_exact() -> None:
    dates = pd.bdate_range("2024-01-01", periods=4)
    returns = pd.DataFrame({"date": dates, "log_return": [0.01, 0.02, 0.03, 0.04]})

    result = ewma_variance_series(
        returns, decay=0.5, initialization_window=3, annualization_days=252
    )

    initial = np.mean(np.square([0.01, 0.02, 0.03]))
    assert np.isclose(result.iloc[0]["value"], 252 * initial)
    assert np.isclose(result.iloc[1]["value"], 252 * (0.5 * initial + 0.5 * 0.04**2))


def test_appending_future_prices_does_not_change_past_rows() -> None:
    early_equity = _equity("2024-05-31")
    full_equity = _equity("2024-06-30")
    early_features = _weekly_features("2024-05-17")
    full_features = _weekly_features("2024-06-21")

    early = construct_targets_and_benchmarks(early_equity, early_features, _config("2024-05-31"))
    full = construct_targets_and_benchmarks(full_equity, full_features, _config())
    overlap = full[full["information_date"] <= early["information_date"].max()].reset_index(
        drop=True
    )

    assert_frame_equal(early, overlap)


def test_forecasts_never_use_source_dates_after_friday() -> None:
    result = construct_targets_and_benchmarks(_equity(), _weekly_features(), _config())

    assert (result["historical_source_date"] <= result["information_date"]).all()
    assert (result["ewma_source_date"] <= result["information_date"]).all()
    assert (result["target_end_inclusive"] > result["information_date"]).all()
    assert (result["target_first_return_date"] > result["information_date"]).all()
    assert (result["target_last_return_date"] <= result["target_end_inclusive"]).all()
    assert (result["target_return_count"] > 0).all()


def test_daily_returns_reject_duplicate_price_dates() -> None:
    equity = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-02", "2024-01-02"]), "value": [100.0, 101.0]}
    )

    try:
        daily_log_returns(equity)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("Expected duplicate dates to be rejected")
