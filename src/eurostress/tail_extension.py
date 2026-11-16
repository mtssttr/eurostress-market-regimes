"""A fixed-design, explicitly exploratory distribution/refitting comparison."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import arch
import numpy as np
import pandas as pd
import scipy
import yaml
from arch import arch_model
from scipy.stats import norm, t

from eurostress.controls import DataQualityError, inspect_frame
from eurostress.risk_validation import (
    christoffersen_independence,
    daily_returns_with_origins,
    kupiec_unconditional_coverage,
)


def tail_losses(
    mean: float, variance: float, probability: float, distribution: str, nu: float | None = None
) -> tuple[float, float]:
    """Analytical VaR/ES for loss=-return, using unit-variance innovations."""
    if not 0 < probability < 0.5 or variance <= 0:
        raise ValueError("Require a positive variance and a lower-tail probability in (0, .5)")
    sigma = np.sqrt(variance)
    if distribution == "normal":
        q = norm.ppf(probability)
        var = -mean - sigma * q
        es = -mean + sigma * norm.pdf(q) / probability
    elif distribution == "t" and nu is not None and nu > 2:
        q = t.ppf(probability, nu)
        scale = np.sqrt((nu - 2) / nu)
        var = -mean - sigma * scale * q
        es = -mean + sigma * scale * (nu + q * q) / (nu - 1) * t.pdf(q, nu) / probability
    else:
        raise ValueError("Use normal or Student t with degrees of freedom > 2")
    return float(var), float(es)


def forecast_comparison(daily: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast t with returns strictly preceding t; refit before the day's return."""
    window = int(settings["estimation"]["trailing_observations"])
    probabilities = settings["evaluation"]["tail_probabilities"]
    dates = pd.to_datetime(daily["return_date"])
    values = daily["log_return"].to_numpy()
    if window < 100 or not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("Require >=100 estimation returns and unique increasing dates")
    if not np.isfinite(values).all() or not all(0 < p < 0.5 for p in probabilities):
        raise ValueError("Require finite returns and valid lower-tail probabilities")
    schedule = settings["estimation"].get("refit_schedule", "first_observation_each_calendar_month")
    if schedule != "first_observation_each_calendar_month":
        raise ValueError("Unsupported refit schedule")
    if settings["estimation"].get("mean", "Constant") != "Constant":
        raise ValueError("Only constant-mean models are specified for this study")
    eligible = np.flatnonzero(
        dates.between(settings["sample"]["evaluation_start"], settings["sample"]["evaluation_end"])
    )
    if len(eligible) < int(settings["evaluation"]["minimum_observations"]):
        raise ValueError("Insufficient evaluation observations")
    if eligible[0] < window:
        raise ValueError("Insufficient pre-evaluation history for the specified window")
    records, fits = [], []
    for distribution in ("normal", "t"):
        for scheme in ("fixed", "monthly"):
            model_name = f"{distribution}_{scheme}"
            params = None
            variance_percent = None
            last_fit_month = None
            for i in eligible:
                month = dates.iloc[i].to_period("M")
                if params is None or (scheme == "monthly" and month != last_fit_month):
                    history = 100 * values[i - window : i]
                    model = arch_model(
                        history,
                        mean="Constant",
                        vol="GARCH",
                        p=1,
                        q=1,
                        dist=distribution,
                        rescale=False,
                    )
                    fitted = model.fit(
                        disp="off",
                        show_warning=False,
                        options={"maxiter": settings["estimation"]["maximum_iterations"]},
                    )
                    if fitted.convergence_flag != 0:
                        raise RuntimeError(f"Nonconvergent fit: {model_name} {dates.iloc[i]}")
                    params = fitted.params.to_dict()
                    variance_percent = float(
                        fitted.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]
                    )
                    last_fit_month = month
                    persistence = params["alpha[1]"] + params["beta[1]"]
                    if persistence > 1 + 1e-8 or not np.isfinite(variance_percent):
                        raise RuntimeError("Invalid fitted variance dynamics")
                    fits.append(
                        {
                            "model": model_name,
                            "forecast_date": dates.iloc[i],
                            "estimation_start": dates.iloc[i - window],
                            "estimation_end": dates.iloc[i - 1],
                            "observations": window,
                            "converged": True,
                            "persistence": persistence,
                            **params,
                        }
                    )
                mean = params["mu"] / 100
                variance = variance_percent / 10000
                for probability in probabilities:
                    var, es = tail_losses(
                        mean, variance, float(probability), distribution, params.get("nu")
                    )
                    loss = -float(values[i])
                    error = loss - var
                    records.append(
                        {
                            "model": model_name,
                            "forecast_origin_date": daily.iloc[i]["forecast_origin_date"],
                            "return_date": dates.iloc[i],
                            "loss": loss,
                            "variance": variance,
                            "tail_probability": probability,
                            "var_loss": var,
                            "es_loss": es,
                            "exception": loss > var,
                            "quantile_loss": (1 - probability) * error
                            if error >= 0
                            else -probability * error,
                        }
                    )
                residual = 100 * values[i] - params["mu"]
                variance_percent = (
                    params["omega"]
                    + params["alpha[1]"] * residual**2
                    + params["beta[1]"] * variance_percent
                )
    return pd.DataFrame(records), pd.DataFrame(fits)


def summarize(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, probability), group in forecasts.groupby(["model", "tail_probability"]):
        hits = group["exception"].to_numpy()
        coverage = kupiec_unconditional_coverage(hits, probability)
        independence = christoffersen_independence(hits)
        tail = group.loc[hits]
        rows.append(
            {
                "model": model,
                "tail_probability": probability,
                "observations": len(group),
                "exceptions": int(hits.sum()),
                "expected_exceptions": len(group) * probability,
                "exception_rate": float(hits.mean()),
                "kupiec_p_value": coverage["p_value"],
                "independence_p_value": independence["p_value"],
                "mean_quantile_loss": float(group["quantile_loss"].mean()),
                "mean_var_loss": float(group["var_loss"].mean()),
                "es_severity_ratio_descriptive": float(tail.loss.mean() / tail.es_loss.mean())
                if len(tail) and tail.es_loss.mean() > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_extension(
    config_path: str | Path, equity_path: str | Path, output_directory: str | Path
) -> dict:
    config_file, equity_file, root = Path(config_path), Path(equity_path), Path(output_directory)
    settings = yaml.safe_load(config_file.read_text())
    if settings.get("classification") != "exploratory_post_hoc":
        raise ValueError("This study must retain its exploratory classification")
    root.mkdir(parents=True, exist_ok=True)
    # Snapshot the design before computing results. This does not undo prior exposure
    # to 2020-2025 outcomes and is not described as a fresh confirmatory test.
    design = {
        "settings": settings,
        "configuration_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(equity_file.read_bytes()).hexdigest(),
    }
    (root / "design_snapshot.json").write_text(json.dumps(design, indent=2) + "\n")
    equity = pd.read_parquet(equity_file)
    findings = inspect_frame(
        equity,
        series="euro_stoxx_50",
        as_of=settings["sample"]["evaluation_end"],
        expected_unit="index_points",
        declared_unit=None,
    )
    if any(x.severity == "ERROR" for x in findings):
        raise DataQualityError("Equity controls failed before exploratory modelling")
    forecasts, fits = forecast_comparison(daily_returns_with_origins(equity), settings)
    # Row-level third-party-derived observations stay in the ignored local data folder.
    local = Path("data/tail_extension")
    local.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(local / "forecasts.parquet", index=False)
    fits.to_parquet(local / "fits.parquet", index=False)
    summary = summarize(forecasts)
    summary.to_csv(root / "comparison.csv", index=False)
    counts = forecasts.groupby(["model", "tail_probability"]).size()
    if counts.nunique() != 1:
        raise RuntimeError("Models were not evaluated on the same observation set")
    primary = summary[
        summary.tail_probability.eq(settings["evaluation"]["primary_tail_probability"])
    ]
    lines = [
        "# Exploratory tail-model comparison",
        "",
        settings["question"],
        "",
        (
            "2020-2025 outcomes had already been examined. These are exploratory comparisons, "
            "not a new untouched holdout or proof of future performance."
        ),
        "",
        "## Fixed design",
        "",
        (
            "GARCH(1,1), constant mean, normal versus unit-variance Student t innovations; "
            "fixed versus first-observation-of-month refitting. Every fit uses exactly the "
            f"preceding {settings['estimation']['trailing_observations']} returns. "
            "Both schedules start with the same estimation window; parameters for fixed "
            "models remain unchanged while conditional variance updates daily. The new fixed "
            "Gaussian comparator uses a different estimation window/library from the original "
            "project baseline and must not be presented as its exact reproduction."
        ),
        "",
        "## 99% VaR comparison",
        "",
        "| Model | Days | Breaches | Rate | Coverage p | Independence p | Quantile loss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples():
        lines.append(
            f"| {row.model} | {row.observations} | {row.exceptions} | "
            f"{row.exception_rate:.2%} | {row.kupiec_p_value:.4f} | "
            f"{row.independence_p_value:.4f} | {row.mean_quantile_loss:.6f} |"
        )
    lines += [
        "",
        (
            "Quantile loss rewards calibrated quantiles rather than simply fewer breaches. "
            "Fewer breaches alone can result from excessively conservative limits. "
            "Coverage and first-order independence p-values use asymptotic reference "
            "distributions; non-rejection does not establish a correct model. "
            "No multiplicity-adjusted superiority claim is made."
        ),
        "",
        (
            "ES severity is reported descriptively at both tail levels in comparison.csv. "
            "It is not a formal ES validation test. The 97.5% ES diagnostic uses 2.5% VaR "
            "exception days. Single-index results do not establish trading-book suitability."
        ),
        "",
        "## Reproduce",
        "",
        "`make tail-extension` (requires existing local equity data).",
        "",
        "References: https://bashtage.github.io/arch/univariate/univariate_volatility_modeling.html",
        "https://doi.org/10.2307/2527341",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))
    metadata = {
        **design,
        "evaluation_days": int(counts.iloc[0]),
        "fits": len(fits),
        "all_fits_converged": bool(fits.converged.all()),
        "versions": {
            "python": platform.python_version(),
            "arch": arch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "forecast_sha256": hashlib.sha256((local / "forecasts.parquet").read_bytes()).hexdigest(),
        "comparison_sha256": hashlib.sha256((root / "comparison.csv").read_bytes()).hexdigest(),
    }
    (root / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
