"""Frozen final-evaluation machinery with an explicit locked-test guard."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss

from eurostress.features import MODEL_FEATURES
from eurostress.models import _hmm_filter, state_diagnostics


def qlike_loss(
    target: np.ndarray, forecast: np.ndarray, positive_floor: float = 1e-12
) -> tuple[np.ndarray, int]:
    """Compute non-negative QLIKE and report numerical-floor applications."""
    y = np.asarray(target, dtype=float)
    h = np.asarray(forecast, dtype=float)
    if y.shape != h.shape:
        raise ValueError("Target and forecast shapes differ")
    if not np.isfinite(y).all() or not np.isfinite(h).all():
        raise ValueError("QLIKE inputs must be finite")
    if (y <= 0).any() or (h <= 0).any():
        raise ValueError("QLIKE inputs must be strictly positive")
    floor_count = int((y < positive_floor).sum() + (h < positive_floor).sum())
    ratio = np.maximum(y, positive_floor) / np.maximum(h, positive_floor)
    return ratio - np.log(ratio) - 1, floor_count


def circular_block_indices(
    sample_size: int, block_length: int, replications: int, seed: int
) -> np.ndarray:
    """Generate circular moving-block bootstrap indices."""
    if sample_size <= 0 or block_length <= 0 or replications <= 0:
        raise ValueError("Bootstrap dimensions must be positive")
    generator = np.random.default_rng(seed)
    block_count = int(np.ceil(sample_size / block_length))
    starts = generator.integers(0, sample_size, size=(replications, block_count))
    offsets = np.arange(block_length)
    blocks = (starts[:, :, None] + offsets[None, None, :]) % sample_size
    return blocks.reshape(replications, -1)[:, :sample_size]


def paired_effect_summary(
    candidate_loss: np.ndarray,
    reference_loss: np.ndarray,
    bootstrap_indices: np.ndarray,
    confidence_level: float,
) -> dict[str, float]:
    """Summarize reference-minus-candidate QLIKE with a percentile interval."""
    candidate = np.asarray(candidate_loss, dtype=float)
    reference = np.asarray(reference_loss, dtype=float)
    if candidate.shape != reference.shape:
        raise ValueError("Paired losses must have equal shapes")
    differences = reference - candidate
    bootstrap_means = differences[bootstrap_indices].mean(axis=1)
    tail = (1 - confidence_level) / 2
    return {
        "candidate_mean_qlike": float(candidate.mean()),
        "reference_mean_qlike": float(reference.mean()),
        "mean_effect": float(differences.mean()),
        "ci_lower": float(np.quantile(bootstrap_means, tail)),
        "ci_upper": float(np.quantile(bootstrap_means, 1 - tail)),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evaluation_config(path: str | Path) -> dict[str, Any]:
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if settings.get("status") != "frozen_pre_test":
        raise ValueError("Evaluation configuration is not frozen_pre_test")
    if settings.get("locked_test_opened") is not False:
        raise ValueError("Evaluation configuration does not describe a sealed test")
    return settings


def freeze_evaluation(
    evaluation_config_path: str | Path,
    output_path: str | Path,
    repository_root: str | Path = ".",
) -> Path:
    """Hash code/artifacts and freeze expected data hashes without reading test rows."""
    root = Path(repository_root).resolve()
    config_path = Path(evaluation_config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    settings = load_evaluation_config(config_path)
    hashes: dict[str, str] = {}
    for relative in settings["critical_files"]:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluation-lock input: {relative}")
        hashes[relative] = _sha256(path)

    feature_manifest = json.loads(
        (root / "data/processed/feature_manifest.json").read_text(encoding="utf-8")
    )
    benchmark_manifest = json.loads(
        (root / "data/processed/benchmark_manifest.json").read_text(encoding="utf-8")
    )
    model_manifest = json.loads(
        (root / "data/models/model_manifest.json").read_text(encoding="utf-8")
    )
    exploratory_manifest = json.loads(
        (root / "data/exploratory/exploratory_manifest.json").read_text(encoding="utf-8")
    )
    lock = {
        "status": "frozen_locked_test_unopened",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_config": str(config_path.relative_to(root)),
        "evaluation_config_sha256": _sha256(config_path),
        "critical_file_hashes": hashes,
        "expected_data_hashes": {
            "data/processed/weekly_features.parquet": feature_manifest["outputs"][
                "weekly_features.parquet"
            ],
            "data/processed/weekly_features_scaled.parquet": feature_manifest["outputs"][
                "weekly_features_scaled.parquet"
            ],
            "data/processed/weekly_targets_benchmarks.parquet": benchmark_manifest["output"][
                "weekly_targets_benchmarks.parquet"
            ],
        },
        "dependency_run_ids": {
            "features": feature_manifest["run_id"],
            "benchmarks": benchmark_manifest["run_id"],
            "confirmatory_models": model_manifest["run_id"],
            "exploratory_models": exploratory_manifest["run_id"],
        },
        "expected_locked_rows": int(settings["sample"]["expected_complete_rows"]),
        "runtime_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "pyyaml": yaml.__version__,
        },
        "locked_test_rows_read_while_freezing": 0,
        "target_performance_metrics_computed_while_freezing": False,
    }
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = root / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return destination


def verify_evaluation_lock(
    lock_path: str | Path, repository_root: str | Path = ".", verify_data: bool = False
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    lock_file = Path(lock_path)
    if not lock_file.is_absolute():
        lock_file = root / lock_file
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    if lock.get("status") != "frozen_locked_test_unopened":
        raise ValueError("Evaluation lock has an invalid status")
    for relative, expected in lock["critical_file_hashes"].items():
        actual = _sha256(root / relative)
        if actual != expected:
            raise ValueError(f"Evaluation-lock hash mismatch: {relative}")
    if verify_data:
        for relative, expected in lock["expected_data_hashes"].items():
            actual = _sha256(root / relative)
            if actual != expected:
                raise ValueError(f"Frozen data hash mismatch: {relative}")
    return lock


def _next_week_return_frame(features: pd.DataFrame) -> pd.DataFrame:
    outcomes = features.loc[:, ["information_date", "equity_return_1w"]].copy()
    outcomes["information_date"] = outcomes["information_date"] - pd.to_timedelta(7, unit="D")
    return outcomes.rename(columns={"equity_return_1w": "next_week_equity_return"})


def _state_tail_mapping(
    probabilities: np.ndarray,
    dates: pd.Series,
    next_returns: pd.DataFrame,
    training_threshold: float,
) -> np.ndarray:
    frame = pd.DataFrame(
        probabilities, columns=[f"state_{index}" for index in range(probabilities.shape[1])]
    )
    frame.insert(0, "information_date", pd.to_datetime(dates).to_numpy())
    aligned = frame.merge(next_returns, on="information_date", how="inner", validate="one_to_one")
    event = (aligned["next_week_equity_return"] < training_threshold).astype(float).to_numpy()
    weights = aligned.filter(like="state_").to_numpy()
    return (weights.T @ event) / weights.sum(axis=0)


def _single_hmm_outputs(
    artifact: dict[str, Any],
    scaled_features: pd.DataFrame,
    next_returns: pd.DataFrame,
    tail_threshold: float,
) -> dict[str, np.ndarray]:
    x = scaled_features.loc[:, artifact["feature_names"]].to_numpy(dtype=float)
    probabilities, predictive_log_density = _hmm_filter(artifact["model"], x)
    training_mask = scaled_features["split"].eq("training").to_numpy()
    state_tail = _state_tail_mapping(
        probabilities[training_mask],
        scaled_features.loc[training_mask, "information_date"],
        next_returns,
        tail_threshold,
    )
    return {
        "variance": probabilities @ np.asarray(artifact["state_variances"]),
        "tail_probability": probabilities @ state_tail,
        "probabilities": probabilities,
        "predictive_log_density": predictive_log_density,
    }


def _ensemble_outputs(
    artifact: dict[str, Any],
    scaled_features: pd.DataFrame,
    next_returns: pd.DataFrame,
    tail_threshold: float,
) -> dict[str, np.ndarray]:
    x = scaled_features.loc[:, artifact["feature_names"]].to_numpy(dtype=float)
    training_mask = scaled_features["split"].eq("training").to_numpy()
    variances: list[np.ndarray] = []
    tail_probabilities: list[np.ndarray] = []
    log_densities: list[np.ndarray] = []
    for member in artifact["members"]:
        probabilities, predictive = _hmm_filter(member["model"], x)
        state_tail = _state_tail_mapping(
            probabilities[training_mask],
            scaled_features.loc[training_mask, "information_date"],
            next_returns,
            tail_threshold,
        )
        variances.append(probabilities @ np.asarray(member["state_variances"]))
        tail_probabilities.append(probabilities @ state_tail)
        log_densities.append(predictive)
    return {
        "variance": np.mean(variances, axis=0),
        "tail_probability": np.mean(tail_probabilities, axis=0),
        "predictive_log_density": np.logaddexp.reduce(log_densities, axis=0)
        - np.log(len(log_densities)),
    }


def _write_final_report(
    output_path: Path,
    comparison_table: pd.DataFrame,
    metrics: dict[str, Any],
    lock_hash: str,
) -> None:
    lines = [
        "# Frozen locked-test evaluation",
        "",
        f"**Evaluation lock SHA-256:** `{lock_hash}`  ",
        "**Original confirmatory cross-asset result:** Not testable; no admissible Phase 4 model.",
        "",
        "## QLIKE comparisons",
        "",
        "| Comparison | Classification | Mean effect | 95% CI | Decision |",
        "|---|---|---:|---:|---|",
    ]
    for row in comparison_table.itertuples(index=False):
        lines.append(
            f"| {row.comparison_id} | {row.classification} | {row.mean_effect:.6f} | "
            f"[{row.ci_lower:.6f}, {row.ci_upper:.6f}] | {row.decision} |"
        )
    lines.extend(
        [
            "",
            "## Secondary metrics",
            "",
            "```json",
            json.dumps(metrics, indent=2),
            "```",
            "",
            "Exploratory intervals are descriptive. This evaluation supports no causal or trading claim.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_locked_test(
    evaluation_config_path: str | Path,
    lock_path: str | Path,
    output_directory: str | Path,
    repository_root: str | Path = ".",
    *,
    unlock_locked_test: bool = False,
    reproduce_frozen_evaluation: bool = False,
) -> Path:
    """Run the frozen evaluation only after an explicit one-time unlock."""
    if not unlock_locked_test and not reproduce_frozen_evaluation:
        raise PermissionError("Locked test remains sealed; explicit unlock is required")
    root = Path(repository_root).resolve()
    settings = load_evaluation_config(root / evaluation_config_path)
    verify_evaluation_lock(lock_path, root, verify_data=True)
    output = root / output_directory
    output.mkdir(parents=True, exist_ok=True)
    opened_marker = output / "locked_test_opened.json"
    if opened_marker.exists() and not reproduce_frozen_evaluation:
        raise RuntimeError("Locked test was already opened; use the frozen reproduction mode")
    if reproduce_frozen_evaluation and not opened_marker.exists():
        raise RuntimeError("Cannot reproduce an evaluation that has not been opened")

    scaled = pd.read_parquet(root / "data/processed/weekly_features_scaled.parquet").sort_values(
        "information_date"
    )
    unscaled = pd.read_parquet(root / "data/processed/weekly_features.parquet").sort_values(
        "information_date"
    )
    targets = pd.read_parquet(
        root / "data/processed/weekly_targets_benchmarks.parquet"
    ).sort_values("information_date")
    next_returns = _next_week_return_frame(unscaled)
    training_next_returns = next_returns.merge(
        scaled.loc[scaled["split"].eq("training"), ["information_date"]],
        on="information_date",
        how="inner",
    )
    tail_threshold = float(
        training_next_returns["next_week_equity_return"].quantile(
            float(settings["secondary_metrics"]["tail_event"]["threshold_quantile"])
        )
    )

    volatility_artifact = joblib.load(root / settings["model_roles"]["confirmatory_secondary"])
    pca_artifact = joblib.load(root / settings["model_roles"]["exploratory_primary"])
    ensemble_artifact = joblib.load(root / settings["model_roles"]["exploratory_robustness"])
    volatility_outputs = _single_hmm_outputs(
        volatility_artifact, scaled, next_returns, tail_threshold
    )
    pca_scaled = scaled.copy()
    pca_x = pca_artifact["pca"].transform(
        scaled.loc[:, pca_artifact["feature_names"]].to_numpy(dtype=float)
    )
    pca_probabilities, pca_predictive = _hmm_filter(pca_artifact["model"], pca_x)
    training_mask = scaled["split"].eq("training").to_numpy()
    pca_state_tail = _state_tail_mapping(
        pca_probabilities[training_mask],
        scaled.loc[training_mask, "information_date"],
        next_returns,
        tail_threshold,
    )
    pca_outputs = {
        "variance": pca_probabilities @ np.asarray(pca_artifact["state_variances"]),
        "tail_probability": pca_probabilities @ pca_state_tail,
        "probabilities": pca_probabilities,
        "predictive_log_density": pca_predictive,
    }
    ensemble_outputs = _ensemble_outputs(
        ensemble_artifact, scaled, next_returns, tail_threshold
    )

    model_frame = pd.DataFrame(
        {
            "information_date": scaled["information_date"],
            "hmm_volatility_only": volatility_outputs["variance"],
            "hmm_pca_exploratory": pca_outputs["variance"],
            "hmm_cross_asset_seed_ensemble": ensemble_outputs["variance"],
            "volatility_hmm_tail_probability": volatility_outputs["tail_probability"],
            "pca_hmm_tail_probability": pca_outputs["tail_probability"],
            "ensemble_hmm_tail_probability": ensemble_outputs["tail_probability"],
            "pca_shock_score": -pca_outputs["predictive_log_density"],
        }
    )
    high_state = pca_artifact["state_labels"].index("higher_training_risk")
    model_frame["pca_high_risk_probability"] = pca_probabilities[:, high_state]
    evaluation = targets[targets["split"].eq(settings["sample"]["required_split_label"])].merge(
        model_frame, on="information_date", how="inner", validate="one_to_one"
    )
    evaluation = evaluation.merge(
        unscaled.loc[:, ["information_date", "new_ciss_external"]],
        on="information_date",
        how="inner",
        validate="one_to_one",
    ).merge(next_returns, on="information_date", how="inner", validate="one_to_one")
    expected_rows = int(settings["sample"]["expected_complete_rows"])
    if len(evaluation) != expected_rows or evaluation.isna().any().any():
        raise ValueError("Locked evaluation does not match the frozen complete-row count")

    forecast_columns = {
        "ewma": "ewma_variance_forecast",
        "historical": "historical_variance_forecast",
        "hmm_volatility_only": "hmm_volatility_only",
        "hmm_pca_exploratory": "hmm_pca_exploratory",
        "hmm_cross_asset_seed_ensemble": "hmm_cross_asset_seed_ensemble",
    }
    losses: dict[str, np.ndarray] = {}
    floor_counts: dict[str, int] = {}
    for name, column in forecast_columns.items():
        losses[name], floor_counts[name] = qlike_loss(
            evaluation["realised_variance"].to_numpy(),
            evaluation[column].to_numpy(),
            float(settings["qlike"]["positive_floor"]),
        )
    bootstrap = settings["bootstrap"]
    main_indices = circular_block_indices(
        len(evaluation),
        int(bootstrap["block_length_weeks"]),
        int(bootstrap["replications"]),
        int(bootstrap["random_seed"]),
    )
    comparison_rows: list[dict[str, Any]] = []
    for comparison in settings["comparisons"]["rows"]:
        summary = paired_effect_summary(
            losses[comparison["candidate"]],
            losses[comparison["reference"]],
            main_indices,
            float(bootstrap["confidence_level"]),
        )
        sensitivities: dict[str, list[float]] = {}
        for block_length in bootstrap["sensitivity_block_lengths_weeks"]:
            indices = circular_block_indices(
                len(evaluation),
                int(block_length),
                int(bootstrap["replications"]),
                int(bootstrap["random_seed"]),
            )
            sensitivity = paired_effect_summary(
                losses[comparison["candidate"]],
                losses[comparison["reference"]],
                indices,
                float(bootstrap["confidence_level"]),
            )
            sensitivities[str(block_length)] = [
                sensitivity["ci_lower"],
                sensitivity["ci_upper"],
            ]
        comparison_rows.append(
            {
                "comparison_id": comparison["id"],
                "candidate": comparison["candidate"],
                "reference": comparison["reference"],
                "classification": comparison["classification"],
                **summary,
                "decision": (
                    "confirmatory_secondary_outperformance"
                    if comparison["classification"] == "secondary_confirmatory"
                    and summary["mean_effect"] > 0
                    and summary["ci_lower"] > 0
                    else (
                        "confirmatory_secondary_outperformance_not_established"
                        if comparison["classification"] == "secondary_confirmatory"
                        else "exploratory_descriptive_only"
                    )
                ),
                "sensitivity_intervals_json": json.dumps(sensitivities),
            }
        )
    comparison_table = pd.DataFrame(comparison_rows)

    test_tail = (
        evaluation["next_week_equity_return"].to_numpy() < tail_threshold
    ).astype(int)
    tail_metrics: dict[str, dict[str, float]] = {}
    for name, column in {
        "volatility_hmm": "volatility_hmm_tail_probability",
        "pca_hmm": "pca_hmm_tail_probability",
        "ensemble_hmm": "ensemble_hmm_tail_probability",
    }.items():
        probabilities = evaluation[column].to_numpy()
        tail_metrics[name] = {
            "brier_score": float(brier_score_loss(test_tail, probabilities)),
            "average_precision": float(average_precision_score(test_tail, probabilities)),
        }

    validation_mask = scaled["split"].eq("validation").to_numpy()
    shock_threshold = float(
        np.quantile(
            -pca_predictive[validation_mask],
            float(settings["secondary_metrics"]["shock_score"]["threshold_quantile"]),
            method=str(settings["secondary_metrics"]["shock_score"]["quantile_method"]),
        )
    )
    shock_flags = evaluation["pca_shock_score"].to_numpy() >= shock_threshold
    test_mask = scaled["split"].eq("locked_test").to_numpy()
    occupancy, durations = state_diagnostics(pca_probabilities[test_mask][:-1])
    ciss_correlation = spearmanr(
        evaluation["pca_high_risk_probability"], evaluation["new_ciss_external"]
    )

    ablation_results: dict[str, float] = {}
    full_pca_loss = losses["hmm_pca_exploratory"].mean()
    for feature in MODEL_FEATURES:
        ablated = pca_scaled.loc[:, pca_artifact["feature_names"]].copy()
        ablated[feature] = 0.0
        transformed = pca_artifact["pca"].transform(ablated.to_numpy(dtype=float))
        probabilities, _ = _hmm_filter(pca_artifact["model"], transformed)
        forecast = probabilities @ np.asarray(pca_artifact["state_variances"])
        forecast_frame = pd.DataFrame(
            {"information_date": scaled["information_date"], "forecast": forecast}
        )
        aligned = evaluation.loc[:, ["information_date", "realised_variance"]].merge(
            forecast_frame, on="information_date", how="left", validate="one_to_one"
        )
        loss, _ = qlike_loss(
            aligned["realised_variance"].to_numpy(),
            aligned["forecast"].to_numpy(),
            float(settings["qlike"]["positive_floor"]),
        )
        ablation_results[feature] = float(loss.mean() - full_pca_loss)

    secondary_metrics = {
        "training_tail_threshold": tail_threshold,
        "test_tail_event_count": int(test_tail.sum()),
        "tail_metrics": tail_metrics,
        "pca_shock_threshold": shock_threshold,
        "pca_shock_count": int(shock_flags.sum()),
        "shock_false_positive_count": int((shock_flags & (test_tail == 0)).sum()),
        "shock_missed_tail_count": int((~shock_flags & (test_tail == 1)).sum()),
        "pca_test_state_occupancy": occupancy,
        "pca_test_median_duration_weeks": durations,
        "pca_high_risk_probability_ciss_spearman": float(ciss_correlation.statistic),
        "pca_high_risk_probability_ciss_pvalue_descriptive": float(ciss_correlation.pvalue),
        "pca_feature_ablation_mean_qlike_change": ablation_results,
        "qlike_floor_counts": floor_counts,
    }

    evaluation.to_parquet(output / "locked_forecasts_and_outcomes.parquet", index=False)
    comparison_table.to_csv(output / "comparison_table.csv", index=False)
    (output / "secondary_metrics.json").write_text(
        json.dumps(secondary_metrics, indent=2) + "\n", encoding="utf-8"
    )
    lock_file = root / lock_path
    lock_hash = _sha256(lock_file)
    _write_final_report(
        output / "final_evaluation_report.md", comparison_table, secondary_metrics, lock_hash
    )
    result_hashes = {
        path.name: _sha256(path)
        for path in (
            output / "locked_forecasts_and_outcomes.parquet",
            output / "comparison_table.csv",
            output / "secondary_metrics.json",
            output / "final_evaluation_report.md",
        )
    }
    if not reproduce_frozen_evaluation:
        opened_marker.write_text(
            json.dumps(
                {
                    "opened_at_utc": datetime.now(UTC).isoformat(),
                    "evaluation_lock_sha256": lock_hash,
                    "result_hashes": result_hashes,
                    "frozen_evaluation_reproducible_only": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return output / "final_evaluation_report.md"
