import numpy as np
import pandas as pd

from eurostress.numerical_validation import run_checks
from eurostress.tail_extension import forecast_comparison


def test_independent_numerical_references_and_known_null():
    result = run_checks()
    assert result["status"] == "PASS"


def test_walk_forward_timing_common_dates_and_current_return_invariance():
    dates = pd.bdate_range("2018-01-01", periods=260)
    daily = pd.DataFrame(
        {
            "return_date": dates,
            "forecast_origin_date": dates - pd.offsets.BDay(),
            "log_return": np.random.default_rng(8).normal(0, 0.01, 260),
        }
    )
    settings = {
        "sample": {
            "evaluation_start": str(dates[220].date()),
            "evaluation_end": str(dates[-1].date()),
        },
        "estimation": {"trailing_observations": 200, "maximum_iterations": 1000},
        "evaluation": {"tail_probabilities": [0.01], "minimum_observations": 20},
    }
    first, fits = forecast_comparison(daily, settings)
    shocked = daily.copy()
    shocked.loc[259, "log_return"] = -0.5
    second, _ = forecast_comparison(shocked, settings)
    a = first[first.return_date.eq(dates[-1])].sort_values("model")
    b = second[second.return_date.eq(dates[-1])].sort_values("model")
    assert np.allclose(a.var_loss, b.var_loss)
    assert np.allclose(a.es_loss, b.es_loss)
    assert (fits.estimation_end < fits.forecast_date).all()
    assert first.groupby("model").size().eq(40).all()
    assert (first.es_loss >= first.var_loss).all()
    assert fits[fits.model.str.endswith("fixed")].groupby("model").size().eq(1).all()
