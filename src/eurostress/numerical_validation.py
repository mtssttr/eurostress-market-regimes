"""Independent reference calculations and a seeded calibration experiment."""

from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path

import arch
import numpy as np
from arch.univariate.volatility import GARCH
from scipy.integrate import quad
from scipy.stats import norm, t

from eurostress.risk_validation import (
    GarchFit,
    _gaussian_tail_forecasts,
    _historical_tail_forecasts,
    christoffersen_independence,
    garch11_forecast_variance,
    kupiec_unconditional_coverage,
)
from eurostress.tail_extension import tail_losses


def run_checks() -> dict:
    rng = np.random.default_rng(20261116)
    fit = GarchFit(0.02, 0.01, 0.08, 0.88, -1.0, True, 800)
    returns = rng.normal(0, 0.01, 800)
    residuals = 100 * returns - fit.mean_percent
    reference = np.zeros(len(returns))
    initial = fit.omega_percent_squared / (1 - fit.alpha - fit.beta)
    GARCH(p=1, q=1).compute_variance(
        np.array([fit.omega_percent_squared, fit.alpha, fit.beta]),
        residuals,
        reference,
        initial,
        np.tile([1e-12, 1e12], (len(returns), 1)),
    )
    ours = garch11_forecast_variance(returns, fit)
    recursion_error = float(np.max(np.abs(ours - reference / 10000)))
    assert recursion_error < 1e-12

    errors = []
    for distribution, nu in (("normal", None), ("t", 5.0), ("t", 10.0)):
        for probability in (0.01, 0.025, 0.05):
            var, es = tail_losses(0.001, 0.0004, probability, distribution, nu)
            if nu is None:
                q = norm.ppf(probability)
                integral = quad(lambda z: z * norm.pdf(z), -np.inf, q)[0]
                implied_probability = norm.cdf((-var - 0.001) / 0.02)
                reference_es = -0.001 - 0.02 * integral / probability
                original_var, original_es = _gaussian_tail_forecasts(
                    np.array([0.001]), np.array([0.0004]), np.array([probability])
                )[probability]
                assert abs(float(original_var[0]) - var) < 1e-10
                assert abs(float(original_es[0]) - reference_es) < 1e-10
            else:
                scale = np.sqrt((nu - 2) / nu)
                q = t.ppf(probability, nu)
                integral = quad(lambda z, df=nu: z * t.pdf(z, df), -np.inf, q)[0]
                implied_probability = t.cdf((-var - 0.001) / (0.02 * scale), nu)
                reference_es = -0.001 - 0.02 * scale * integral / probability
            assert abs(implied_probability - probability) < 1e-10
            assert abs(es - reference_es) < 1e-10
            errors.append(abs(es - reference_es))

    historical_var, historical_es = _historical_tail_forecasts(
        np.array([-0.05, -0.025, 0.0, 0.02, 0.05, -0.90]), 5, np.array([0.25])
    )[0.25]
    assert abs(historical_var[-1] - 0.025) < 1e-12
    assert abs(historical_es[-1] - 0.0375) < 1e-12

    hits = np.array([0] * 90 + [1] * 10, dtype=bool)
    ll0 = 10 * math.log(0.05) + 90 * math.log(0.95)
    ll1 = 10 * math.log(0.10) + 90 * math.log(0.90)
    kupiec_error = abs(kupiec_unconditional_coverage(hits, 0.05)["statistic"] - 2 * (ll1 - ll0))
    assert kupiec_error < 1e-10
    sequence = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0], dtype=bool)
    counts = np.zeros((2, 2), dtype=int)
    for previous, current in pairwise(sequence):
        counts[int(previous), int(current)] += 1
    pooled = float(counts[:, 1].sum() / counts.sum())

    def term(count: int, probability: float) -> float:
        return count * math.log(probability) if count else 0.0

    null_ll = sum(
        term(int(counts[a, b]), pooled if b else 1 - pooled) for a in range(2) for b in range(2)
    )
    alternative_ll = 0.0
    for row in counts:
        p = row[1] / row.sum()
        alternative_ll += term(int(row[0]), 1 - p) + term(int(row[1]), p)
    independence_error = abs(
        christoffersen_independence(sequence)["statistic"] - 2 * (alternative_ll - null_ll)
    )
    assert independence_error < 1e-10

    # The reference model has exactly known Gaussian quantiles. This validates
    # test behaviour under a specified null, not calibration on actual markets.
    simulations = rng.standard_normal((1000, 2000))
    breaches = simulations < norm.ppf(0.05)
    rejection_rate = float(
        np.mean([kupiec_unconditional_coverage(row, 0.05)["p_value"] < 0.05 for row in breaches])
    )
    assert 0.02 < rejection_rate < 0.08
    return {
        "status": "PASS",
        "seed": 20261116,
        "arch_version": arch.__version__,
        "garch_max_absolute_variance_error": recursion_error,
        "es_max_absolute_quadrature_error": float(max(errors)),
        "kupiec_reference_statistic_error": kupiec_error,
        "independence_reference_statistic_error": independence_error,
        "simulation_replications": 1000,
        "simulation_observations_per_replication": 2000,
        "simulation_tail_probability": 0.05,
        "test_significance": 0.05,
        "simulation_coverage_rejection_rate": rejection_rate,
        "scope": "Numerical reference checks and specified-null simulation; not market validation.",
    }


def main() -> None:
    root = Path("reports/numerical_validation")
    root.mkdir(parents=True, exist_ok=True)
    result = run_checks()
    (root / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Numerical validation",
        "",
        "All assertions passed.",
        "",
        (
            "Checks cover a GARCH variance path against arch with identical parameters and "
            "initial variance, analytical normal/Student t tails against numerical quadrature, "
            "and independent likelihood calculations for coverage and independence."
        ),
        "",
        (
            "The seeded Gaussian experiment checks empirical rejection frequency under "
            "a known null. It does not establish calibration on financial data."
        ),
        "",
        "```json",
        json.dumps(result, indent=2),
        "```",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
