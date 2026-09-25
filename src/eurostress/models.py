"""Training/validation-only mixture and hidden-state model selection."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

from eurostress.config import ProjectConfig
from eurostress.features import MODEL_FEATURES

MODEL_VERSION = "1.0.0"
FAMILY_FEATURES = {
    "gmm_cross_asset": MODEL_FEATURES,
    "hmm_volatility_only": ("equity_log_rv_20d",),
    "hmm_cross_asset": MODEL_FEATURES,
}


@dataclass(frozen=True)
class ModelBuildResult:
    run_id: str
    training_rows: int
    validation_rows: int
    candidate_count: int
    admissible_candidate_count: int
    selected_models: dict[str, dict[str, int] | None]
    locked_test_rows_read: int = 0
    target_performance_metrics_computed: bool = False


def forward_filter(
    observations: np.ndarray,
    start_probability: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    diagonal_covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return causal filtered probabilities and one-step predictive log densities."""
    x = np.asarray(observations, dtype=float)
    start = np.asarray(start_probability, dtype=float)
    transition = np.asarray(transition_matrix, dtype=float)
    centers = np.asarray(means, dtype=float)
    covariances = np.asarray(diagonal_covariances, dtype=float)
    if covariances.ndim == 3:
        covariances = np.diagonal(covariances, axis1=1, axis2=2)
    if x.ndim != 2 or centers.shape[1] != x.shape[1]:
        raise ValueError("Observation and emission dimensions do not match")
    if (covariances <= 0).any():
        raise ValueError("Emission variances must be positive")

    log_emissions = -0.5 * (
        x.shape[1] * np.log(2 * np.pi)
        + np.log(covariances).sum(axis=1)[None, :]
        + (((x[:, None, :] - centers[None, :, :]) ** 2) / covariances[None, :, :]).sum(
            axis=2
        )
    )
    filtered = np.empty((len(x), len(start)))
    predictive_log_density = np.empty(len(x))
    prior = start
    for index in range(len(x)):
        log_weights = np.log(np.clip(prior, 1e-300, None)) + log_emissions[index]
        normalization = logsumexp(log_weights)
        predictive_log_density[index] = normalization
        filtered[index] = np.exp(log_weights - normalization)
        prior = filtered[index] @ transition
    return filtered, predictive_log_density


def state_diagnostics(probabilities: np.ndarray) -> tuple[list[float], list[float]]:
    """Compute filtered occupancy and hard-filtered median run duration by state."""
    if probabilities.ndim != 2 or len(probabilities) == 0:
        raise ValueError("State probabilities must be a non-empty matrix")
    occupancy = probabilities.mean(axis=0).tolist()
    hard_states = probabilities.argmax(axis=1)
    durations: dict[int, list[int]] = {state: [] for state in range(probabilities.shape[1])}
    run_state = int(hard_states[0])
    run_length = 1
    for state in hard_states[1:]:
        state = int(state)
        if state == run_state:
            run_length += 1
        else:
            durations[run_state].append(run_length)
            run_state = state
            run_length = 1
    durations[run_state].append(run_length)
    medians = [float(np.median(durations[state])) if durations[state] else 0.0 for state in durations]
    return occupancy, medians


def training_state_variances(
    probabilities: np.ndarray, information_dates: pd.Series, targets: pd.DataFrame
) -> np.ndarray:
    """Map states to next-week variance using training rows only."""
    probability_frame = pd.DataFrame(
        probabilities,
        columns=[f"state_{index}" for index in range(probabilities.shape[1])],
    )
    probability_frame.insert(0, "information_date", pd.to_datetime(information_dates).to_numpy())
    aligned = probability_frame.merge(
        targets.loc[:, ["information_date", "realised_variance"]],
        on="information_date",
        how="inner",
        validate="one_to_one",
    )
    if aligned.empty:
        raise ValueError("No training targets align with state probabilities")
    weights = aligned.filter(like="state_").to_numpy()
    outcomes = aligned["realised_variance"].to_numpy()
    denominators = weights.sum(axis=0)
    if (denominators <= 0).any():
        raise ValueError("A state has no training probability mass")
    mapped = (weights.T @ outcomes) / denominators
    if not np.isfinite(mapped).all() or (mapped <= 0).any():
        raise ValueError("State variance mapping is not positive and finite")
    return mapped


def select_candidate(
    candidates: pd.DataFrame, tie_tolerance_per_observation: float
) -> pd.Series | None:
    """Select highest validation density, resolving near-ties toward fewer states."""
    admissible = candidates[candidates["admissible"]].copy()
    if admissible.empty:
        return None
    best_score = admissible["validation_loglik_per_observation"].max()
    tied = admissible[
        admissible["validation_loglik_per_observation"]
        >= best_score - tie_tolerance_per_observation
    ]
    return tied.sort_values(
        ["state_count", "validation_loglik_per_observation"], ascending=[True, False]
    ).iloc[0]


def _hmm_filter(model: GaussianHMM, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return forward_filter(
        observations,
        model.startprob_,
        model.transmat_,
        model.means_,
        model.covars_,
    )


def _fit_hmm_seed(
    x_train: np.ndarray, x_all: np.ndarray, state_count: int, seed: int, settings: dict[str, Any]
) -> tuple[GaussianHMM | None, dict[str, Any]]:
    model = GaussianHMM(
        n_components=state_count,
        covariance_type="diag",
        min_covar=float(settings["covariance_floor"]),
        n_iter=int(settings["maximum_iterations"]),
        tol=float(settings["convergence_tolerance"]),
        random_state=seed,
        implementation="log",
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train)
        converged = bool(model.monitor_.converged)
        train_loglik = float(model.score(x_train))
        _, predictive = _hmm_filter(model, x_all)
        validation_predictive = predictive[len(x_train) :]
        validation_loglik = float(validation_predictive.sum())
        finite = np.isfinite([train_loglik, validation_loglik]).all()
        converged = converged and bool(finite)
        return (
            model if converged else None,
            {
                "seed": seed,
                "converged": converged,
                "training_loglik": train_loglik,
                "validation_loglik": validation_loglik,
                "validation_loglik_per_observation": float(validation_predictive.mean()),
                "error": "",
            },
        )
    except Exception as error:  # noqa: BLE001  # pragma: no cover
        return None, {
            "seed": seed,
            "converged": False,
            "training_loglik": np.nan,
            "validation_loglik": np.nan,
            "validation_loglik_per_observation": np.nan,
            "error": type(error).__name__,
        }


def _fit_gmm_seed(
    x_train: np.ndarray, x_validation: np.ndarray, state_count: int, seed: int, settings: dict[str, Any]
) -> tuple[GaussianMixture | None, dict[str, Any]]:
    model = GaussianMixture(
        n_components=state_count,
        covariance_type="diag",
        reg_covar=float(settings["covariance_floor"]),
        max_iter=int(settings["maximum_iterations"]),
        tol=float(settings["convergence_tolerance"]),
        n_init=1,
        random_state=seed,
        init_params="kmeans",
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train)
        train_loglik = float(model.score_samples(x_train).sum())
        validation_densities = model.score_samples(x_validation)
        validation_loglik = float(validation_densities.sum())
        converged = bool(model.converged_) and bool(
            np.isfinite([train_loglik, validation_loglik]).all()
        )
        return (
            model if converged else None,
            {
                "seed": seed,
                "converged": converged,
                "training_loglik": train_loglik,
                "validation_loglik": validation_loglik,
                "validation_loglik_per_observation": float(validation_densities.mean()),
                "error": "",
            },
        )
    except Exception as error:  # noqa: BLE001  # pragma: no cover
        return None, {
            "seed": seed,
            "converged": False,
            "training_loglik": np.nan,
            "validation_loglik": np.nan,
            "validation_loglik_per_observation": np.nan,
            "error": type(error).__name__,
        }


def _fit_family(
    family: str,
    x_train: np.ndarray,
    x_validation: np.ndarray,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, Any]]:
    candidates: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    best_models: dict[int, Any] = {}
    initial_seed = int(settings["deterministic_seed_start"])
    seeds = range(initial_seed, initial_seed + int(settings["deterministic_initializations"]))

    for state_count in settings["state_counts"]:
        fitted: list[tuple[Any, dict[str, Any]]] = []
        x_all = np.vstack([x_train, x_validation])
        for seed in seeds:
            if family.startswith("hmm_"):
                model, seed_result = _fit_hmm_seed(
                    x_train, x_all, int(state_count), seed, settings
                )
            else:
                model, seed_result = _fit_gmm_seed(
                    x_train, x_validation, int(state_count), seed, settings
                )
            row = {"family": family, "state_count": int(state_count), **seed_result}
            seed_rows.append(row)
            if model is not None:
                fitted.append((model, row))

        converged_fraction = len(fitted) / int(settings["deterministic_initializations"])
        if not fitted:
            candidates.append(
                {
                    "family": family,
                    "state_count": int(state_count),
                    "best_seed": None,
                    "validation_loglik": np.nan,
                    "validation_loglik_per_observation": np.nan,
                    "converged_seed_fraction": converged_fraction,
                    "validation_loglik_iqr_per_observation": np.nan,
                    "minimum_validation_occupancy": np.nan,
                    "minimum_validation_median_duration": np.nan,
                    "admissible": False,
                    "failure_reasons": "no_converged_seed",
                }
            )
            continue

        best_model, best_row = max(fitted, key=lambda item: item[1]["training_loglik"])
        best_models[int(state_count)] = best_model
        validation_scores = np.array(
            [item[1]["validation_loglik_per_observation"] for item in fitted]
        )
        validation_iqr = float(np.quantile(validation_scores, 0.75) - np.quantile(validation_scores, 0.25))
        if family.startswith("hmm_"):
            probabilities, _ = _hmm_filter(best_model, x_all)
            validation_probabilities = probabilities[len(x_train) :]
        else:
            validation_probabilities = best_model.predict_proba(x_validation)
        occupancy, durations = state_diagnostics(validation_probabilities)

        reasons: list[str] = []
        if converged_fraction < float(settings["minimum_converged_seed_fraction"]):
            reasons.append("converged_seed_fraction")
        if validation_iqr > float(
            settings["maximum_validation_loglik_iqr_per_observation"]
        ):
            reasons.append("seed_instability")
        if min(occupancy) < float(settings["minimum_state_occupancy"]):
            reasons.append("state_occupancy")
        if family.startswith("hmm_") and min(durations) < float(
            settings["minimum_median_duration_weeks"]
        ):
            reasons.append("state_duration")
        candidates.append(
            {
                "family": family,
                "state_count": int(state_count),
                "best_seed": int(best_row["seed"]),
                "validation_loglik": float(best_row["validation_loglik"]),
                "validation_loglik_per_observation": float(
                    best_row["validation_loglik_per_observation"]
                ),
                "converged_seed_fraction": converged_fraction,
                "validation_loglik_iqr_per_observation": validation_iqr,
                "minimum_validation_occupancy": float(min(occupancy)),
                "minimum_validation_median_duration": float(min(durations)),
                "admissible": not reasons,
                "failure_reasons": ";".join(reasons),
            }
        )
    return candidates, seed_rows, best_models


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_run_id(config_path: Path, feature_manifest: Path, benchmark_manifest: Path) -> str:
    digest = hashlib.sha256()
    digest.update(MODEL_VERSION.encode())
    digest.update(config_path.read_bytes())
    for path in (feature_manifest, benchmark_manifest):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        digest.update(metadata["run_id"].encode())
    return f"models-{digest.hexdigest()[:16]}"


def build_models(
    config: ProjectConfig,
    config_path: str | Path,
    processed_directory: str | Path,
    model_directory: str | Path,
    report_directory: str | Path,
) -> ModelBuildResult:
    """Fit and select model families without loading locked-test rows."""
    processed = Path(processed_directory)
    model_path = Path(model_directory)
    reports = Path(report_directory)
    diagnostics = reports / "model_selection"
    model_path.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(parents=True, exist_ok=True)
    feature_manifest = processed / "feature_manifest.json"
    benchmark_manifest = processed / "benchmark_manifest.json"
    required = (
        processed / "weekly_features_scaled.parquet",
        processed / "weekly_targets_benchmarks.parquet",
        feature_manifest,
        benchmark_manifest,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 4 input: {path}")

    allowed_splits = ["training", "validation"]
    features = pd.read_parquet(
        required[0], filters=[("split", "in", allowed_splits)]
    ).sort_values("information_date")
    targets = pd.read_parquet(
        required[1], filters=[("split", "in", allowed_splits)]
    ).sort_values("information_date")
    validation_end = pd.Timestamp(config.splits["validation_end"])
    if features["information_date"].max() > validation_end:
        raise ValueError("Phase 4 loaded a row after the validation cutoff")
    if targets["information_date"].max() > validation_end:
        raise ValueError("Phase 4 loaded a target after the validation cutoff")

    train_features = features[features["split"] == "training"].reset_index(drop=True)
    validation_features = features[features["split"] == "validation"].reset_index(drop=True)
    train_targets = targets[targets["split"] == "training"].reset_index(drop=True)
    settings = config.models
    candidate_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    family_models: dict[str, dict[int, Any]] = {}

    for family, feature_names in FAMILY_FEATURES.items():
        x_train = train_features.loc[:, feature_names].to_numpy(dtype=float)
        x_validation = validation_features.loc[:, feature_names].to_numpy(dtype=float)
        candidates, seeds, best_models = _fit_family(
            family, x_train, x_validation, settings
        )
        candidate_rows.extend(candidates)
        seed_rows.extend(seeds)
        family_models[family] = best_models

    candidate_frame = pd.DataFrame(candidate_rows)
    seed_frame = pd.DataFrame(seed_rows)
    candidate_frame.to_csv(diagnostics / "phase4_candidate_diagnostics.csv", index=False)
    seed_frame.to_csv(diagnostics / "phase4_seed_diagnostics.csv", index=False)

    run_id = _model_run_id(Path(config_path), feature_manifest, benchmark_manifest)
    selected_metadata: dict[str, dict[str, int] | None] = {}
    forecast_frames: list[pd.DataFrame] = []
    artifact_hashes: dict[str, str] = {}
    for family in FAMILY_FEATURES:
        (model_path / f"{family}.joblib").unlink(missing_ok=True)
    (model_path / "validation_forecasts.parquet").unlink(missing_ok=True)
    for family, feature_names in FAMILY_FEATURES.items():
        selection = select_candidate(
            candidate_frame[candidate_frame["family"] == family],
            float(settings["tie_tolerance_loglik_per_observation"]),
        )
        if selection is None:
            selected_metadata[family] = None
            continue
        state_count = int(selection["state_count"])
        seed = int(selection["best_seed"])
        model = family_models[family][state_count]
        x_train = train_features.loc[:, feature_names].to_numpy(dtype=float)
        x_validation = validation_features.loc[:, feature_names].to_numpy(dtype=float)
        if family.startswith("hmm_"):
            probabilities, _ = _hmm_filter(model, np.vstack([x_train, x_validation]))
            train_probabilities = probabilities[: len(x_train)]
            validation_probabilities = probabilities[len(x_train) :]
        else:
            train_probabilities = model.predict_proba(x_train)
            validation_probabilities = model.predict_proba(x_validation)
        state_variances = training_state_variances(
            train_probabilities, train_features["information_date"], train_targets
        )
        forecasts = validation_probabilities @ state_variances
        forecast_frames.append(
            pd.DataFrame(
                {
                    "information_date": validation_features["information_date"],
                    "family": family,
                    "state_count": state_count,
                    "seed": seed,
                    "variance_forecast": forecasts,
                }
            )
        )
        artifact = {
            "family": family,
            "model_version": MODEL_VERSION,
            "run_id": run_id,
            "model": model,
            "feature_names": list(feature_names),
            "state_variances": state_variances,
            "training_last_information_date": train_features["information_date"].max(),
            "validation_last_information_date": validation_features["information_date"].max(),
            "state_count": state_count,
            "seed": seed,
        }
        artifact_path = model_path / f"{family}.joblib"
        joblib.dump(artifact, artifact_path)
        artifact_hashes[artifact_path.name] = _sha256(artifact_path)
        selected_metadata[family] = {"state_count": state_count, "seed": seed}

    validation_forecasts = pd.concat(forecast_frames, ignore_index=True)
    if not np.isfinite(validation_forecasts["variance_forecast"]).all():
        raise ValueError("Validation forecasts contain non-finite values")
    if (validation_forecasts["variance_forecast"] <= 0).any():
        raise ValueError("Validation forecasts must be positive")
    validation_forecasts.to_parquet(model_path / "validation_forecasts.parquet", index=False)
    artifact_hashes["validation_forecasts.parquet"] = _sha256(
        model_path / "validation_forecasts.parquet"
    )

    result = ModelBuildResult(
        run_id=run_id,
        training_rows=len(train_features),
        validation_rows=len(validation_features),
        candidate_count=len(candidate_frame),
        admissible_candidate_count=int(candidate_frame["admissible"].sum()),
        selected_models=selected_metadata,
    )
    manifest = {
        **asdict(result),
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "configuration_sha256": _sha256(Path(config_path)),
        "feature_manifest_sha256": _sha256(feature_manifest),
        "benchmark_manifest_sha256": _sha256(benchmark_manifest),
        "settings": dict(settings),
        "artifacts": artifact_hashes,
    }
    (model_path / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return result
