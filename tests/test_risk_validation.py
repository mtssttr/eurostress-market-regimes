import numpy as np
import pandas as pd

from eurostress.risk_validation import (
    GarchFit,
    _gaussian_tail_forecasts,
    _historical_tail_forecasts,
    attach_regime_information,
    christoffersen_independence,
    ewma_forecast_variance,
    fit_garch11_gaussian,
    garch11_forecast_variance,
    kupiec_unconditional_coverage,
)


def test_historical_var_does_not_use_current_return() -> None:
    baseline = np.array([0.01, -0.01, 0.02, -0.02, 0.005])
    shocked = baseline.copy()
    shocked[-1] = -0.50

    first = _historical_tail_forecasts(baseline, 4, np.array([0.25]))[0.25]
    second = _historical_tail_forecasts(shocked, 4, np.array([0.25]))[0.25]

    assert np.isclose(first[0][-1], second[0][-1])
    assert np.isclose(first[1][-1], second[1][-1])


def test_gaussian_expected_shortfall_is_not_smaller_than_var() -> None:
    forecasts = _gaussian_tail_forecasts(
        np.array([0.0, 0.001]), np.array([0.0001, 0.0004]), np.array([0.01, 0.05])
    )

    for var_loss, es_loss in forecasts.values():
        assert np.all(es_loss >= var_loss)


def test_ewma_forecast_excludes_current_return() -> None:
    baseline = np.array([0.01, -0.01, 0.02, -0.02, 0.01, 0.03])
    shocked = baseline.copy()
    shocked[-1] = -0.50

    first = ewma_forecast_variance(baseline, 0.94, 4)
    second = ewma_forecast_variance(shocked, 0.94, 4)

    assert np.isclose(first[-1], second[-1])


def test_garch_fit_is_stationary_and_forecast_positive() -> None:
    rng = np.random.default_rng(4)
    returns = rng.normal(0, 0.01, 800)
    fit = fit_garch11_gaussian(returns, maximum_iterations=500)
    forecasts = garch11_forecast_variance(returns, fit)

    assert fit.converged
    assert fit.alpha >= 0
    assert fit.beta >= 0
    assert fit.alpha + fit.beta < 1
    assert np.isfinite(forecasts).all()
    assert (forecasts > 0).all()


def test_garch_current_shock_changes_only_later_forecast() -> None:
    fit = GarchFit(0.0, 0.01, 0.08, 0.88, -1.0, True, 100)
    baseline = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
    shocked = baseline.copy()
    shocked[-1] = -0.50

    first = garch11_forecast_variance(baseline, fit)
    second = garch11_forecast_variance(shocked, fit)

    assert np.isclose(first[-1], second[-1])


def test_coverage_tests_return_valid_probabilities() -> None:
    hits = np.array([0] * 95 + [1] * 5, dtype=bool)
    kupiec = kupiec_unconditional_coverage(hits, 0.05)
    independence = christoffersen_independence(hits)

    assert np.isclose(kupiec["statistic"], 0.0)
    assert 0 <= kupiec["p_value"] <= 1
    assert 0 <= independence["p_value"] <= 1
    assert sum(independence[key] for key in ["n00", "n01", "n10", "n11"]) == 99


def test_regime_join_uses_forecast_origin_not_return_date() -> None:
    forecasts = pd.DataFrame(
        {
            "forecast_origin_date": pd.to_datetime(["2024-01-04", "2024-01-05"]),
            "return_date": pd.to_datetime(["2024-01-05", "2024-01-08"]),
        }
    )
    regimes = pd.DataFrame(
        {
            "information_date": pd.to_datetime(["2023-12-29", "2024-01-05"]),
            "pca_high_risk_probability": [0.1, 0.9],
            "new_ciss_external": [0.2, 0.8],
        }
    )

    result = attach_regime_information(forecasts, regimes, 0.5)

    assert result.loc[0, "regime"] == "lower-risk regime"
    assert result.loc[1, "regime"] == "higher-risk regime"


def test_regime_join_marks_dates_before_first_weekly_estimate_unavailable() -> None:
    forecasts = pd.DataFrame(
        {
            "forecast_origin_date": pd.to_datetime(["2023-12-28"]),
            "return_date": pd.to_datetime(["2023-12-29"]),
        }
    )
    regimes = pd.DataFrame(
        {
            "information_date": pd.to_datetime(["2024-01-05"]),
            "pca_high_risk_probability": [0.9],
            "new_ciss_external": [0.8],
        }
    )

    result = attach_regime_information(forecasts, regimes, 0.5)

    assert result.loc[0, "regime"] == "regime unavailable"
