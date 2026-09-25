"""Post-hoc market-risk model validation across EuroStress regimes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import xlogy
from scipy.stats import chi2, norm

RISK_VALIDATION_VERSION = "1.0.0"
MODEL_ORDER = (
    "historical_simulation",
    "rolling_gaussian",
    "ewma_gaussian",
    "garch11_gaussian",
)
MODEL_LABELS = {
    "historical_simulation": "Historical simulation",
    "rolling_gaussian": "Rolling Gaussian",
    "ewma_gaussian": "EWMA Gaussian",
    "garch11_gaussian": "GARCH(1,1) Gaussian",
}


@dataclass(frozen=True)
class GarchFit:
    mean_percent: float
    omega_percent_squared: float
    alpha: float
    beta: float
    log_likelihood: float
    converged: bool
    observations: int


@dataclass(frozen=True)
class RiskValidationResult:
    evaluation_start: str
    evaluation_end: str
    evaluation_days: int
    model_count: int
    tail_probability_count: int
    high_risk_days: int
    report_path: str
    figure_path: str


def load_risk_validation_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the compact post-hoc risk-validation specification."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Risk-validation configuration root must be a mapping")
    required = {"sample", "models", "evaluation", "stress_windows", "reporting"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Missing risk-validation sections: {sorted(missing)}")

    probabilities = np.asarray(raw["evaluation"]["tail_probabilities"], dtype=float)
    if probabilities.size == 0 or np.any((probabilities <= 0) | (probabilities >= 0.5)):
        raise ValueError("Tail probabilities must lie strictly between zero and 0.5")
    if float(raw["models"]["ewma_lambda"]) <= 0 or float(
        raw["models"]["ewma_lambda"]
    ) >= 1:
        raise ValueError("EWMA lambda must lie strictly between zero and one")
    return raw


def daily_returns_with_origins(equity: pd.DataFrame) -> pd.DataFrame:
    """Create close-to-close returns and the last date known before each return."""
    required = {"date", "value"}
    if not required <= set(equity.columns):
        raise ValueError(f"Equity data are missing columns: {sorted(required - set(equity.columns))}")
    prices = equity.loc[:, ["date", "value"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="raise").dt.tz_localize(None)
    prices["value"] = pd.to_numeric(prices["value"], errors="raise")
    prices = prices.sort_values("date").reset_index(drop=True)
    if prices.empty or prices["date"].duplicated().any():
        raise ValueError("Equity prices must be non-empty with unique dates")
    if prices["value"].isna().any() or (prices["value"] <= 0).any():
        raise ValueError("Equity prices must be finite and strictly positive")

    result = pd.DataFrame(
        {
            "forecast_origin_date": prices["date"].shift(1),
            "return_date": prices["date"],
            "log_return": np.log(prices["value"] / prices["value"].shift(1)),
        }
    ).dropna()
    result["loss"] = -result["log_return"]
    if not np.isfinite(result[["log_return", "loss"]].to_numpy()).all():
        raise ValueError("Daily returns contain non-finite values")
    return result.reset_index(drop=True)


def _garch_parameters(theta: np.ndarray, stationarity_cap: float) -> tuple[float, ...]:
    mean_percent = float(theta[0])
    omega = float(np.exp(np.clip(theta[1], -30.0, 20.0)))
    logits = np.array([0.0, theta[2], theta[3]], dtype=float)
    logits -= logits.max()
    weights = np.exp(logits) / np.exp(logits).sum()
    alpha = float(stationarity_cap * weights[1])
    beta = float(stationarity_cap * weights[2])
    return mean_percent, omega, alpha, beta


def _garch_variance_path(
    returns_percent: np.ndarray,
    mean_percent: float,
    omega: float,
    alpha: float,
    beta: float,
    initial_variance: float,
) -> np.ndarray:
    variances = np.full(len(returns_percent), np.nan, dtype=float)
    variances[0] = max(float(initial_variance), 1e-10)
    residuals = returns_percent - mean_percent
    for index in range(1, len(returns_percent)):
        variances[index] = (
            omega + alpha * residuals[index - 1] ** 2 + beta * variances[index - 1]
        )
    return np.maximum(variances, 1e-10)


def fit_garch11_gaussian(
    returns: np.ndarray,
    *,
    stationarity_cap: float = 0.999,
    maximum_iterations: int = 2000,
) -> GarchFit:
    """Fit a stationary Gaussian GARCH(1,1) by maximum likelihood."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) < 100 or not np.isfinite(values).all():
        raise ValueError("GARCH fitting requires at least 100 finite returns")
    if not 0 < stationarity_cap < 1:
        raise ValueError("GARCH stationarity cap must lie strictly between zero and one")

    returns_percent = 100.0 * values
    sample_variance = max(float(np.var(returns_percent, ddof=1)), 1e-8)
    alpha_start = 0.05
    beta_start = 0.90
    remainder = stationarity_cap - alpha_start - beta_start
    if remainder <= 0:
        raise ValueError("Stationarity cap is too small for the initial GARCH values")
    theta_start = np.array(
        [
            float(np.mean(returns_percent)),
            np.log(sample_variance * (1 - alpha_start - beta_start)),
            np.log((alpha_start / stationarity_cap) / (remainder / stationarity_cap)),
            np.log((beta_start / stationarity_cap) / (remainder / stationarity_cap)),
        ]
    )

    def objective(theta: np.ndarray) -> float:
        mean_percent, omega, alpha, beta = _garch_parameters(theta, stationarity_cap)
        variances = _garch_variance_path(
            returns_percent,
            mean_percent,
            omega,
            alpha,
            beta,
            sample_variance,
        )
        residuals = returns_percent - mean_percent
        return float(
            0.5 * np.sum(np.log(2 * np.pi) + np.log(variances) + residuals**2 / variances)
        )

    fitted = minimize(
        objective,
        theta_start,
        method="L-BFGS-B",
        bounds=[(-5.0, 5.0), (-25.0, 10.0), (-12.0, 12.0), (-12.0, 12.0)],
        options={"maxiter": int(maximum_iterations), "ftol": 1e-12},
    )
    mean_percent, omega, alpha, beta = _garch_parameters(
        np.asarray(fitted.x), stationarity_cap
    )
    return GarchFit(
        mean_percent=mean_percent,
        omega_percent_squared=omega,
        alpha=alpha,
        beta=beta,
        log_likelihood=-float(fitted.fun),
        converged=bool(fitted.success),
        observations=len(values),
    )


def garch11_forecast_variance(returns: np.ndarray, fit: GarchFit) -> np.ndarray:
    """One-step variance for each return, using only returns available the day before."""
    values = np.asarray(returns, dtype=float)
    returns_percent = 100.0 * values
    unconditional = fit.omega_percent_squared / max(1 - fit.alpha - fit.beta, 1e-6)
    percent_variance = _garch_variance_path(
        returns_percent,
        fit.mean_percent,
        fit.omega_percent_squared,
        fit.alpha,
        fit.beta,
        unconditional,
    )
    return percent_variance / 10_000.0


def ewma_forecast_variance(
    returns: np.ndarray, decay: float, initialization_window: int
) -> np.ndarray:
    """One-step EWMA variance aligned so the current return is never used in its forecast."""
    values = np.asarray(returns, dtype=float)
    if len(values) <= initialization_window:
        raise ValueError("Insufficient returns for EWMA initialization")
    forecasts = np.full(len(values), np.nan, dtype=float)
    forecasts[initialization_window] = float(np.mean(values[:initialization_window] ** 2))
    for index in range(initialization_window + 1, len(values)):
        forecasts[index] = (
            decay * forecasts[index - 1] + (1 - decay) * values[index - 1] ** 2
        )
    return forecasts


def _historical_tail_forecasts(
    returns: np.ndarray, window: int, probabilities: np.ndarray
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    values = np.asarray(returns, dtype=float)
    output = {
        float(probability): (np.full(len(values), np.nan), np.full(len(values), np.nan))
        for probability in probabilities
    }
    for index in range(window, len(values)):
        history = values[index - window : index]
        for probability in probabilities:
            probability = float(probability)
            quantile = float(np.quantile(history, probability, method="linear"))
            tail = history[history <= quantile]
            var_values, es_values = output[probability]
            var_values[index] = max(-quantile, 0.0)
            es_values[index] = max(-float(np.mean(tail)), var_values[index])
    return output


def _gaussian_tail_forecasts(
    means: np.ndarray, variances: np.ndarray, probabilities: np.ndarray
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    output: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    standard_deviations = np.sqrt(np.maximum(np.asarray(variances, dtype=float), 0.0))
    mean_values = np.asarray(means, dtype=float)
    for probability in probabilities:
        probability = float(probability)
        z_value = float(norm.ppf(probability))
        var_loss = -(mean_values + standard_deviations * z_value)
        es_loss = -mean_values + standard_deviations * float(norm.pdf(z_value)) / probability
        output[probability] = (np.maximum(var_loss, 0.0), np.maximum(es_loss, var_loss))
    return output


def build_tail_risk_forecasts(
    returns: pd.DataFrame,
    config: dict[str, Any],
    garch_fit: GarchFit,
) -> pd.DataFrame:
    """Create point-in-time one-day VaR and ES forecasts for four conventional models."""
    values = returns["log_return"].to_numpy(dtype=float)
    probabilities = np.asarray(config["evaluation"]["tail_probabilities"], dtype=float)
    historical_window = int(config["models"]["historical_window_trading_days"])
    gaussian_window = int(config["models"]["gaussian_window_trading_days"])

    historical = _historical_tail_forecasts(values, historical_window, probabilities)
    shifted = pd.Series(values).shift(1)
    rolling_means = shifted.rolling(gaussian_window, min_periods=gaussian_window).mean().to_numpy()
    rolling_variances = (
        shifted.rolling(gaussian_window, min_periods=gaussian_window).var(ddof=1).to_numpy()
    )
    rolling_gaussian = _gaussian_tail_forecasts(
        rolling_means, rolling_variances, probabilities
    )
    ewma_variances = ewma_forecast_variance(
        values,
        float(config["models"]["ewma_lambda"]),
        int(config["models"]["ewma_initialization_trading_days"]),
    )
    ewma_gaussian = _gaussian_tail_forecasts(
        np.zeros(len(values)), ewma_variances, probabilities
    )
    garch_variances = garch11_forecast_variance(values, garch_fit)
    garch_gaussian = _gaussian_tail_forecasts(
        np.full(len(values), garch_fit.mean_percent / 100.0),
        garch_variances,
        probabilities,
    )
    forecasts_by_model = {
        "historical_simulation": historical,
        "rolling_gaussian": rolling_gaussian,
        "ewma_gaussian": ewma_gaussian,
        "garch11_gaussian": garch_gaussian,
    }

    frames: list[pd.DataFrame] = []
    base = returns.loc[
        :, ["forecast_origin_date", "return_date", "log_return", "loss"]
    ].reset_index(drop=True)
    for model in MODEL_ORDER:
        for probability in probabilities:
            var_loss, es_loss = forecasts_by_model[model][float(probability)]
            frame = base.copy()
            frame["model"] = model
            frame["tail_probability"] = float(probability)
            frame["var_loss"] = var_loss
            frame["es_loss"] = es_loss
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result = result.dropna(subset=["var_loss", "es_loss"])
    if (result["es_loss"] < result["var_loss"]).any():
        raise AssertionError("Expected shortfall must be at least as large as VaR")
    return result


def attach_regime_information(
    forecasts: pd.DataFrame,
    weekly_regimes: pd.DataFrame,
    probability_threshold: float,
) -> pd.DataFrame:
    """Attach the last Friday regime known at each forecast origin."""
    required = {"information_date", "pca_high_risk_probability", "new_ciss_external"}
    if not required <= set(weekly_regimes.columns):
        raise ValueError(f"Weekly regimes are missing columns: {sorted(required - set(weekly_regimes))}")
    origins = forecasts.loc[:, ["forecast_origin_date"]].drop_duplicates().sort_values(
        "forecast_origin_date"
    )
    weekly = weekly_regimes.loc[:, sorted(required)].copy()
    weekly["information_date"] = pd.to_datetime(weekly["information_date"])
    weekly = weekly.sort_values("information_date")
    attached = pd.merge_asof(
        origins,
        weekly,
        left_on="forecast_origin_date",
        right_on="information_date",
        direction="backward",
        allow_exact_matches=True,
    )
    result = forecasts.merge(attached, on="forecast_origin_date", validate="many_to_one")
    result["regime"] = np.select(
        [
            result["pca_high_risk_probability"].isna(),
            result["pca_high_risk_probability"] >= probability_threshold,
        ],
        ["regime unavailable", "higher-risk regime"],
        default="lower-risk regime",
    )
    return result


def assign_stress_windows(
    forecasts: pd.DataFrame, windows: list[dict[str, str]]
) -> pd.DataFrame:
    result = forecasts.copy()
    result["stress_window"] = "outside named windows"
    for window in windows:
        start = pd.Timestamp(window["start"])
        end = pd.Timestamp(window["end"])
        mask = result["return_date"].between(start, end)
        result.loc[mask, "stress_window"] = str(window["name"])
    return result


def kupiec_unconditional_coverage(
    exceedances: np.ndarray, tail_probability: float
) -> dict[str, float]:
    """Kupiec likelihood-ratio test for the unconditional exception rate."""
    hits = np.asarray(exceedances, dtype=bool)
    observations = len(hits)
    exceptions = int(hits.sum())
    if observations == 0:
        raise ValueError("Coverage testing requires observations")
    observed = exceptions / observations
    null_log_likelihood = float(
        xlogy(exceptions, tail_probability)
        + xlogy(observations - exceptions, 1 - tail_probability)
    )
    alternative_log_likelihood = float(
        xlogy(exceptions, observed) + xlogy(observations - exceptions, 1 - observed)
    )
    statistic = max(0.0, -2 * (null_log_likelihood - alternative_log_likelihood))
    return {"statistic": statistic, "p_value": float(chi2.sf(statistic, 1))}


def christoffersen_independence(exceedances: np.ndarray) -> dict[str, float | int]:
    """Christoffersen likelihood-ratio test for first-order exception independence."""
    hits = np.asarray(exceedances, dtype=int)
    if len(hits) < 2:
        raise ValueError("Independence testing requires at least two observations")
    previous = hits[:-1]
    current = hits[1:]
    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())
    total = n00 + n01 + n10 + n11
    pooled = (n01 + n11) / total
    probability_after_zero = n01 / (n00 + n01) if n00 + n01 else 0.0
    probability_after_one = n11 / (n10 + n11) if n10 + n11 else 0.0
    null_log_likelihood = float(
        xlogy(n01 + n11, pooled) + xlogy(n00 + n10, 1 - pooled)
    )
    alternative_log_likelihood = float(
        xlogy(n01, probability_after_zero)
        + xlogy(n00, 1 - probability_after_zero)
        + xlogy(n11, probability_after_one)
        + xlogy(n10, 1 - probability_after_one)
    )
    statistic = max(0.0, -2 * (null_log_likelihood - alternative_log_likelihood))
    return {
        "statistic": statistic,
        "p_value": float(chi2.sf(statistic, 1)),
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
    }


def summarize_backtests(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Summarize coverage, independence, conditional coverage and ES severity."""
    rows: list[dict[str, Any]] = []
    for (model, probability), group in forecasts.groupby(
        ["model", "tail_probability"], sort=False
    ):
        group = group.sort_values("return_date")
        hits = group["loss"].to_numpy() > group["var_loss"].to_numpy()
        kupiec = kupiec_unconditional_coverage(hits, float(probability))
        independence = christoffersen_independence(hits)
        conditional_statistic = float(kupiec["statistic"] + independence["statistic"])
        hit_group = group.loc[hits]
        realised_tail_loss = float(hit_group["loss"].mean()) if hits.any() else np.nan
        forecast_tail_loss = float(hit_group["es_loss"].mean()) if hits.any() else np.nan
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "tail_probability": float(probability),
                "observations": len(group),
                "exceptions": int(hits.sum()),
                "expected_exceptions": float(len(group) * probability),
                "exception_rate": float(hits.mean()),
                "kupiec_lr": float(kupiec["statistic"]),
                "kupiec_p_value": float(kupiec["p_value"]),
                "independence_lr": float(independence["statistic"]),
                "independence_p_value": float(independence["p_value"]),
                "conditional_coverage_lr": conditional_statistic,
                "conditional_coverage_p_value": float(chi2.sf(conditional_statistic, 2)),
                "mean_var_loss": float(group["var_loss"].mean()),
                "mean_es_loss": float(group["es_loss"].mean()),
                "realised_tail_loss": realised_tail_loss,
                "forecast_es_on_exceptions": forecast_tail_loss,
                "es_severity_ratio": realised_tail_loss / forecast_tail_loss
                if hits.any() and forecast_tail_loss > 0
                else np.nan,
                "n00": int(independence["n00"]),
                "n01": int(independence["n01"]),
                "n10": int(independence["n10"]),
                "n11": int(independence["n11"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_groups(
    forecasts: pd.DataFrame, group_column: str, tail_probability: float
) -> pd.DataFrame:
    """Describe exception coverage within regimes or named stress windows."""
    selected = forecasts[np.isclose(forecasts["tail_probability"], tail_probability)].copy()
    selected["exception"] = selected["loss"] > selected["var_loss"]
    rows: list[dict[str, Any]] = []
    for (model, group_name), group in selected.groupby(["model", group_column], sort=False):
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                group_column: group_name,
                "observations": len(group),
                "exceptions": int(group["exception"].sum()),
                "expected_exceptions": float(len(group) * tail_probability),
                "exception_rate": float(group["exception"].mean()),
                "coverage_ratio": float(group["exception"].mean() / tail_probability),
                "mean_var_loss": float(group["var_loss"].mean()),
                "mean_es_loss": float(group["es_loss"].mean()),
                "mean_realised_loss": float(group["loss"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _plot_validation(
    forecasts: pd.DataFrame,
    tail_probability: float,
    rolling_window: int,
    coverage_path: Path,
    regime_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "historical_simulation": "#8C6BB1",
        "rolling_gaussian": "#DD8452",
        "ewma_gaussian": "#2A9D8F",
        "garch11_gaussian": "#277DA1",
    }
    selected = forecasts[np.isclose(forecasts["tail_probability"], tail_probability)].copy()
    first_model = selected[selected["model"].eq(MODEL_ORDER[0])].sort_values("return_date")

    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    axes[0].plot(
        first_model["return_date"],
        100 * first_model["loss"],
        color="#202936",
        linewidth=0.7,
        alpha=0.65,
        label="Daily loss",
    )
    for model in MODEL_ORDER:
        group = selected[selected["model"].eq(model)].sort_values("return_date")
        axes[0].plot(
            group["return_date"],
            100 * group["var_loss"],
            color=colors[model],
            linewidth=1.15,
            label=MODEL_LABELS[model],
        )
    axes[0].set_title("One-day 99% VaR forecasts and realised EURO STOXX 50 losses")
    axes[0].set_ylabel("Loss / VaR (% log return)")
    axes[0].legend(ncol=3, fontsize=8, frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    for model in MODEL_ORDER:
        group = selected[selected["model"].eq(model)].sort_values("return_date").copy()
        group["exception"] = (group["loss"] > group["var_loss"]).astype(float)
        rolling = group["exception"].rolling(rolling_window, min_periods=60).mean()
        axes[1].plot(
            group["return_date"],
            100 * rolling,
            color=colors[model],
            linewidth=1.4,
            label=MODEL_LABELS[model],
        )
    axes[1].axhline(100 * tail_probability, color="#C8524F", linestyle="--", linewidth=1.2)
    axes[1].set_title(f"Rolling {rolling_window}-day exception rate")
    axes[1].set_ylabel("Exceptions (%)")
    axes[1].set_xlabel("Return date")
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(coverage_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    regime_summary = summarize_groups(selected, "regime", tail_probability)
    pivot = regime_summary.pivot(
        index="model", columns="regime", values="exception_rate"
    ).reindex(MODEL_ORDER)
    figure, axis = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    x_values = np.arange(len(MODEL_ORDER))
    width = 0.34
    lower = 100 * pivot.get("lower-risk regime", pd.Series(index=MODEL_ORDER)).to_numpy()
    higher = 100 * pivot.get("higher-risk regime", pd.Series(index=MODEL_ORDER)).to_numpy()
    axis.bar(x_values - width / 2, lower, width, color="#8EC6C5", label="Lower-risk regime")
    axis.bar(x_values + width / 2, higher, width, color="#E7A84B", label="Higher-risk regime")
    axis.axhline(100 * tail_probability, color="#C8524F", linestyle="--", linewidth=1.2)
    axis.set_xticks(x_values, [MODEL_LABELS[model] for model in MODEL_ORDER], rotation=12)
    axis.set_ylabel("99% VaR exception rate (%)")
    axis.set_title("99% VaR exception rates differed across EuroStress regimes")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(regime_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> list[str]:
    header_labels = {
        "model_label": "Model",
        "tail_probability": "Tail probability",
        "exception_rate": "Exception rate",
        "expected_exceptions": "Expected exceptions",
        "kupiec_p_value": "Kupiec p-value",
        "independence_p_value": "Independence p-value",
        "conditional_coverage_p_value": "Conditional-coverage p-value",
        "mean_var_loss": "Mean VaR loss",
        "mean_es_loss": "Mean ES loss",
        "forecast_es_on_exceptions": "Forecast ES on exceptions",
        "es_severity_ratio": "ES severity ratio",
    }
    headers = [
        header_labels.get(column, column.replace("_", " ").title()) for column in columns
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(columns)]
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        values: list[str] = []
        for column, value in zip(columns, row, strict=True):
            if pd.isna(value):
                values.append("—")
            elif column.endswith("_p_value") and float(value) < 0.001:
                values.append("<0.001")
            elif column in formats:
                values.append(format(value, formats[column]))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_report(
    path: Path,
    config: dict[str, Any],
    fit: GarchFit,
    backtests: pd.DataFrame,
    regimes: pd.DataFrame,
    windows: pd.DataFrame,
) -> None:
    primary_var = float(config["evaluation"]["primary_var_tail_probability"])
    primary_es = float(config["evaluation"]["primary_es_tail_probability"])
    primary = backtests[np.isclose(backtests["tail_probability"], primary_var)].copy()
    primary["model_label"] = pd.Categorical(
        primary["model_label"], [MODEL_LABELS[model] for model in MODEL_ORDER], ordered=True
    )
    primary = primary.sort_values("model_label")
    es_primary = backtests[np.isclose(backtests["tail_probability"], primary_es)].copy()
    es_primary["model_label"] = pd.Categorical(
        es_primary["model_label"],
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        ordered=True,
    )
    es_primary = es_primary.sort_values("model_label")
    regime_primary = regimes.copy()
    regime_primary["model_label"] = pd.Categorical(
        regime_primary["model_label"],
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        ordered=True,
    )
    regime_primary = regime_primary.sort_values(["model_label", "regime"])
    named_windows = windows[windows["stress_window"].ne("outside named windows")].copy()
    named_windows["model_label"] = pd.Categorical(
        named_windows["model_label"],
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        ordered=True,
    )
    named_windows = named_windows.sort_values(["stress_window", "model_label"])

    lines = [
        "# Market-risk validation extension",
        "",
        "**Classification:** Post-hoc extension; the original sealed EuroStress result is unchanged.",
        "",
        "## Question",
        "",
        "> How reliable are conventional market-risk models across financial-stress regimes?",
        "",
        (
            "Daily one-step forecasts compare 250-day historical simulation, a 250-day Gaussian "
            "model, RiskMetrics-style EWMA volatility and a Gaussian GARCH(1,1). Every forecast "
            "uses information available by the preceding trading close. Parameters for GARCH are "
            "estimated only through 31 December 2019 and then held fixed through 2020–2025."
        ),
        "",
        "## Primary 99% VaR backtest",
        "",
    ]
    lines.extend(
        _markdown_table(
            primary,
            [
                "model_label",
                "exceptions",
                "expected_exceptions",
                "exception_rate",
                "kupiec_p_value",
                "independence_p_value",
                "conditional_coverage_p_value",
            ],
            {
                "expected_exceptions": ".1f",
                "exception_rate": ".3%",
                "kupiec_p_value": ".3f",
                "independence_p_value": ".3f",
                "conditional_coverage_p_value": ".3f",
            },
        )
    )
    lines.extend(
        [
            "",
            "## Key findings",
            "",
            (
                "- All four models produced more 99% VaR exceptions than the 15.1 expected under "
                "correct coverage. Historical simulation was closest with 26 exceptions, but its "
                "Kupiec p-value of 0.011 still rejects correct unconditional coverage at the 5% "
                "level."
            ),
            (
                "- GARCH reduced the COVID-window exception count to three, compared with six for "
                "EWMA, eight for historical simulation and eleven for the rolling Gaussian model. "
                "Its full-sample exception rate remained too high, but its Christoffersen "
                "independence p-value of 0.794 did not indicate first-order clustering."
            ),
            (
                "- Historical and rolling-Gaussian exception rates rose in the higher-risk regime. "
                "EWMA and GARCH showed the reverse descriptive pattern because their volatility "
                "forecasts increased after shocks; the weekly regime label can also arrive after a "
                "sudden daily loss. This is evidence for conditional model diagnostics, not for a "
                "causal regime effect."
            ),
            "",
            (
                "Kupiec tests unconditional coverage. Christoffersen tests first-order exception "
                "independence. The conditional-coverage statistic combines both. The ES severity "
                "ratio is realised mean exception loss divided by forecast ES on exception days; "
                "it is a descriptive diagnostic, not presented as a standalone formal ES test."
            ),
            "",
            "## 97.5% Expected Shortfall diagnostics",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            es_primary,
            [
                "model_label",
                "exceptions",
                "expected_exceptions",
                "exception_rate",
                "mean_es_loss",
                "realised_tail_loss",
                "es_severity_ratio",
            ],
            {
                "expected_exceptions": ".1f",
                "exception_rate": ".3%",
                "mean_es_loss": ".3%",
                "realised_tail_loss": ".3%",
                "es_severity_ratio": ".2f",
            },
        )
    )
    lines.extend(
        [
            "",
            (
                "The 97.5% level matches the one-tailed confidence level used for Expected "
                "Shortfall in the Basel market-risk framework. A severity ratio above one means "
                "realised exception losses were larger on average than the model's ES forecast. "
                "This ratio is descriptive; formal ES validation requires a joint VaR/ES testing "
                "framework and is left as a clearly identified extension."
            ),
            "",
            "## Coverage by EuroStress regime",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            regime_primary,
            [
                "model_label",
                "regime",
                "observations",
                "exceptions",
                "exception_rate",
                "coverage_ratio",
            ],
            {"exception_rate": ".3%", "coverage_ratio": ".2f"},
        )
    )
    lines.extend(["", "## Named stress-window diagnostics", ""])
    lines.extend(
        _markdown_table(
            named_windows,
            [
                "stress_window",
                "model_label",
                "observations",
                "exceptions",
                "exception_rate",
                "mean_var_loss",
            ],
            {"exception_rate": ".3%", "mean_var_loss": ".3%"},
        )
    )
    lines.extend(
        [
            "",
            "## GARCH fit",
            "",
            f"- Mean daily return: {fit.mean_percent / 100:.5%}",
            f"- Omega: {fit.omega_percent_squared:.6f} percent-squared",
            f"- Alpha: {fit.alpha:.4f}",
            f"- Beta: {fit.beta:.4f}",
            f"- Persistence: {fit.alpha + fit.beta:.4f}",
            f"- Pre-2020 estimation observations: {fit.observations:,}",
            f"- Optimizer converged: {'yes' if fit.converged else 'no'}",
            "",
            "## Interpretation limits",
            "",
            (
                "This is a deliberately contained validation exercise on one equity index. It is "
                "not a bank trading-book capital model, does not implement liquidity-horizon "
                "scaling, and does not establish that the exploratory regime labels cause model "
                "failure. Named stress windows and regime-conditional results are descriptive. "
                "The first evaluation day is excluded from regime splits because no prior Friday "
                "state estimate existed. Raw daily observations and row-level forecasts remain "
                "local and are not redistributed."
            ),
            "",
            "## Methodological references",
            "",
            (
                "- Bollerslev, T. (1986), *Generalized Autoregressive Conditional "
                "Heteroskedasticity*, Journal of Econometrics 31(3), 307–327. "
                "https://doi.org/10.1016/0304-4076(86)90063-1"
            ),
            (
                "- Kupiec, P. (1995), *Techniques for Verifying the Accuracy of Risk Measurement "
                "Models*, Federal Reserve FEDS 95-24. "
                "https://www.fedinprint.org/item/fedgfe/34596/original"
            ),
            (
                "- Christoffersen, P. (1998), *Evaluating Interval Forecasts*, International "
                "Economic Review 39(4), 841–862. https://doi.org/10.2307/2527341"
            ),
            (
                "- McNeil, A. and Frey, R. (2000), *Estimation of Tail-Related Risk Measures for "
                "Heteroscedastic Financial Time Series*, Journal of Empirical Finance 7, 271–300. "
                "https://doi.org/10.1016/S0927-5398(00)00012-8"
            ),
            (
                "- Basel Committee on Banking Supervision, *Basel Framework*, MAR33: Internal "
                "models approach. https://www.bis.org/baselframework/BaselFramework.pdf"
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_risk_validation(
    *,
    config_path: str | Path,
    equity_path: str | Path,
    regime_path: str | Path,
    output_directory: str | Path,
    report_directory: str | Path,
) -> RiskValidationResult:
    """Run the contained market-risk extension and write aggregate public results."""
    config_file = Path(config_path)
    equity_file = Path(equity_path)
    regime_file = Path(regime_path)
    output = Path(output_directory)
    reports = Path(report_directory)
    figures = reports / "figures"
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    config = load_risk_validation_config(config_file)
    equity = pd.read_parquet(equity_file)
    regimes = pd.read_parquet(regime_file)
    daily = daily_returns_with_origins(equity)
    estimation_end = pd.Timestamp(config["sample"]["estimation_end"])
    estimation_returns = daily.loc[
        daily["return_date"] <= estimation_end, "log_return"
    ].to_numpy()
    fit = fit_garch11_gaussian(
        estimation_returns,
        stationarity_cap=float(config["models"]["garch_stationarity_cap"]),
        maximum_iterations=int(config["models"]["garch_maximum_iterations"]),
    )
    if not fit.converged:
        raise RuntimeError("GARCH optimization did not converge")

    forecasts = build_tail_risk_forecasts(daily, config, fit)
    evaluation_start = pd.Timestamp(config["sample"]["evaluation_start"])
    evaluation_end = pd.Timestamp(config["sample"]["evaluation_end"])
    forecasts = forecasts[forecasts["return_date"].between(evaluation_start, evaluation_end)]
    forecasts = attach_regime_information(
        forecasts,
        regimes,
        float(config["evaluation"]["regime_probability_threshold"]),
    )
    forecasts = assign_stress_windows(forecasts, list(config["stress_windows"]))
    forecasts["exception"] = forecasts["loss"] > forecasts["var_loss"]

    backtests = summarize_backtests(forecasts)
    primary_var = float(config["evaluation"]["primary_var_tail_probability"])
    regime_summary = summarize_groups(
        forecasts[forecasts["regime"].ne("regime unavailable")], "regime", primary_var
    )
    window_summary = summarize_groups(forecasts, "stress_window", primary_var)

    forecast_path = output / "daily_risk_forecasts.parquet"
    manifest_path = output / "risk_validation_manifest.json"
    backtest_path = reports / "risk_validation_backtests.csv"
    regime_summary_path = reports / "risk_validation_regime_breakdown.csv"
    window_summary_path = reports / "risk_validation_stress_windows.csv"
    report_path = reports / "risk_validation_report.md"
    coverage_figure_path = figures / "risk_validation_coverage.png"
    regime_figure_path = figures / "risk_validation_regimes.png"
    forecasts.to_parquet(forecast_path, index=False)
    backtests.to_csv(backtest_path, index=False)
    regime_summary.to_csv(regime_summary_path, index=False)
    window_summary.to_csv(window_summary_path, index=False)
    _plot_validation(
        forecasts,
        primary_var,
        int(config["evaluation"]["rolling_coverage_window_trading_days"]),
        coverage_figure_path,
        regime_figure_path,
    )
    _write_report(
        report_path,
        config,
        fit,
        backtests,
        regime_summary,
        window_summary,
    )

    unique_days = forecasts.loc[:, ["return_date"]].drop_duplicates()
    high_risk_days = forecasts.loc[
        forecasts["regime"].eq("higher-risk regime"), ["return_date"]
    ].drop_duplicates()
    result = RiskValidationResult(
        evaluation_start=unique_days["return_date"].min().date().isoformat(),
        evaluation_end=unique_days["return_date"].max().date().isoformat(),
        evaluation_days=len(unique_days),
        model_count=len(MODEL_ORDER),
        tail_probability_count=len(config["evaluation"]["tail_probabilities"]),
        high_risk_days=len(high_risk_days),
        report_path=str(report_path),
        figure_path=str(coverage_figure_path),
    )
    manifest = {
        "risk_validation_version": RISK_VALIDATION_VERSION,
        "classification": config["reporting"]["classification"],
        "configuration_sha256": _sha256(config_file),
        "equity_input_sha256": _sha256(equity_file),
        "regime_input_sha256": _sha256(regime_file),
        "garch_fit": asdict(fit),
        "result": asdict(result),
        "outputs": {
            forecast_path.name: _sha256(forecast_path),
            backtest_path.name: _sha256(backtest_path),
            regime_summary_path.name: _sha256(regime_summary_path),
            window_summary_path.name: _sha256(window_summary_path),
            report_path.name: _sha256(report_path),
            coverage_figure_path.name: _sha256(coverage_figure_path),
            regime_figure_path.name: _sha256(regime_figure_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return result
