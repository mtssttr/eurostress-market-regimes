"""Explicitly exploratory cross-asset HMM stabilization experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.special import logsumexp
from sklearn.decomposition import PCA

from eurostress.config import ProjectConfig
from eurostress.features import MODEL_FEATURES
from eurostress.models import (
    _fit_family,
    _fit_hmm_seed,
    _hmm_filter,
    select_candidate,
    state_diagnostics,
    training_state_variances,
)

EXPLORATORY_VERSION = "1.0.0"


@dataclass(frozen=True)
class ExploratoryBuildResult:
    run_id: str
    training_rows: int
    validation_rows: int
    pca_selected: dict[str, int] | None
    ensemble_selected: dict[str, Any] | None
    locked_test_rows_read: int = 0
    target_performance_metrics_computed: bool = False
    classification: str = "exploratory"


def load_exploratory_config(path: str | Path) -> dict[str, Any]:
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if settings.get("status") != "exploratory":
        raise ValueError("Phase 4B configuration must be labelled exploratory")
    if settings.get("locked_test_access") != "prohibited":
        raise ValueError("Phase 4B must prohibit locked-test access")
    return settings


def ensemble_log_density(member_log_densities: np.ndarray) -> np.ndarray:
    """Equal-weight mixture log density across ensemble members."""
    values = np.asarray(member_log_densities, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Member log densities must be a non-empty matrix")
    return logsumexp(values, axis=0) - np.log(values.shape[0])


def elite_training_members(
    seed_rows: list[dict[str, Any]], training_rows: int, tolerance_per_observation: float
) -> set[int]:
    """Choose the maximum-likelihood basin using training likelihood only."""
    converged = [row for row in seed_rows if row["converged"]]
    if not converged:
        return set()
    best = max(float(row["training_loglik"]) / training_rows for row in converged)
    return {
        int(row["seed"])
        for row in converged
        if float(row["training_loglik"]) / training_rows >= best - tolerance_per_observation
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_id(
    exploratory_config: Path, feature_manifest: Path, benchmark_manifest: Path
) -> str:
    digest = hashlib.sha256()
    digest.update(EXPLORATORY_VERSION.encode())
    digest.update(exploratory_config.read_bytes())
    for path in (feature_manifest, benchmark_manifest):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        digest.update(metadata["run_id"].encode())
    return f"exploratory-{digest.hexdigest()[:16]}"


def _fit_seed_ensemble(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    shared: dict[str, Any],
    ensemble_settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, list[tuple[Any, dict[str, Any]]]]]:
    candidate_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    qualified_models: dict[int, list[tuple[Any, dict[str, Any]]]] = {}
    seed_start = int(shared["deterministic_seed_start"])
    seeds = range(seed_start, seed_start + int(shared["deterministic_initializations"]))
    x_all = np.vstack([x_train, x_validation])

    for raw_state_count in shared["state_counts"]:
        state_count = int(raw_state_count)
        fitted: list[tuple[Any, dict[str, Any]]] = []
        for seed in seeds:
            model, row = _fit_hmm_seed(x_train, x_all, state_count, seed, shared)
            if model is not None:
                fitted.append((model, row))
        elite_seeds = elite_training_members(
            [row for _, row in fitted],
            training_rows=len(x_train),
            tolerance_per_observation=float(
                ensemble_settings["elite_training_loglik_tolerance_per_observation"]
            ),
        )
        qualified: list[tuple[Any, dict[str, Any]]] = []
        member_log_densities: list[np.ndarray] = []
        for model, row in fitted:
            probabilities, predictive = _hmm_filter(model, x_all)
            occupancy, durations = state_diagnostics(probabilities[len(x_train) :])
            is_elite = int(row["seed"]) in elite_seeds
            state_quality_pass = (
                min(occupancy)
                >= float(ensemble_settings["qualification"]["minimum_state_occupancy"])
                and min(durations)
                >= float(
                    ensemble_settings["qualification"]["minimum_median_duration_weeks"]
                )
            )
            member_rows.append(
                {
                    "state_count": state_count,
                    "seed": int(row["seed"]),
                    "converged": bool(row["converged"]),
                    "training_loglik_per_observation": float(row["training_loglik"])
                    / len(x_train),
                    "validation_loglik_per_observation": float(
                        row["validation_loglik_per_observation"]
                    ),
                    "elite_training_basin": is_elite,
                    "minimum_validation_occupancy": float(min(occupancy)),
                    "minimum_validation_median_duration": float(min(durations)),
                    "state_quality_pass": state_quality_pass,
                    "qualified_member": is_elite and state_quality_pass,
                }
            )
            if is_elite and state_quality_pass:
                qualified.append((model, row))
                member_log_densities.append(predictive[len(x_train) :])

        minimum_members = int(ensemble_settings["minimum_qualified_members"])
        operational = len(qualified) >= minimum_members
        if qualified:
            mixture_density = ensemble_log_density(np.vstack(member_log_densities))
            validation_loglik = float(mixture_density.sum())
            validation_per_observation = float(mixture_density.mean())
        else:
            validation_loglik = np.nan
            validation_per_observation = np.nan
        candidate_rows.append(
            {
                "family": "hmm_cross_asset_seed_ensemble",
                "state_count": state_count,
                "converged_seeds": len(fitted),
                "elite_training_seeds": len(elite_seeds),
                "qualified_members": len(qualified),
                "member_seeds": ";".join(str(row["seed"]) for _, row in qualified),
                "validation_loglik": validation_loglik,
                "validation_loglik_per_observation": validation_per_observation,
                "admissible": operational,
                "failure_reasons": "" if operational else "insufficient_qualified_members",
            }
        )
        qualified_models[state_count] = qualified
    return pd.DataFrame(candidate_rows), pd.DataFrame(member_rows), qualified_models


def build_exploratory_models(
    core_config: ProjectConfig,
    exploratory_config_path: str | Path,
    processed_directory: str | Path,
    output_directory: str | Path,
    report_directory: str | Path,
) -> ExploratoryBuildResult:
    """Run PCA and seed-ensemble stabilization without loading locked-test rows."""
    exploratory_path = Path(exploratory_config_path)
    settings = load_exploratory_config(exploratory_path)
    processed = Path(processed_directory)
    output = Path(output_directory)
    reports = Path(report_directory)
    diagnostics = reports / "exploratory"
    output.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(parents=True, exist_ok=True)
    feature_manifest = processed / "feature_manifest.json"
    benchmark_manifest = processed / "benchmark_manifest.json"
    feature_path = processed / "weekly_features_scaled.parquet"
    target_path = processed / "weekly_targets_benchmarks.parquet"
    for path in (feature_manifest, benchmark_manifest, feature_path, target_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 4B input: {path}")

    allowed_splits = ["training", "validation"]
    features = pd.read_parquet(
        feature_path, filters=[("split", "in", allowed_splits)]
    ).sort_values("information_date")
    targets = pd.read_parquet(
        target_path, filters=[("split", "in", allowed_splits)]
    ).sort_values("information_date")
    validation_end = pd.Timestamp(core_config.splits["validation_end"])
    if features["information_date"].max() > validation_end:
        raise ValueError("Phase 4B loaded features after the validation cutoff")
    if targets["information_date"].max() > validation_end:
        raise ValueError("Phase 4B loaded targets after the validation cutoff")

    train = features[features["split"] == "training"].reset_index(drop=True)
    validation = features[features["split"] == "validation"].reset_index(drop=True)
    train_targets = targets[targets["split"] == "training"].reset_index(drop=True)
    x_train = train.loc[:, MODEL_FEATURES].to_numpy(dtype=float)
    x_validation = validation.loc[:, MODEL_FEATURES].to_numpy(dtype=float)
    shared = settings["shared"]

    pca = PCA(
        n_components=int(settings["pca_hmm"]["components"]),
        svd_solver=str(settings["pca_hmm"]["solver"]),
    )
    x_train_pca = pca.fit_transform(x_train)
    x_validation_pca = pca.transform(x_validation)
    pca_candidates, pca_seeds, pca_models = _fit_family(
        "hmm_pca_exploratory", x_train_pca, x_validation_pca, shared
    )
    pca_candidate_frame = pd.DataFrame(pca_candidates)
    pca_seed_frame = pd.DataFrame(pca_seeds)
    pca_candidate_frame.to_csv(diagnostics / "phase4b_pca_candidates.csv", index=False)
    pca_seed_frame.to_csv(diagnostics / "phase4b_pca_seeds.csv", index=False)

    ensemble_candidates, ensemble_members, ensemble_models = _fit_seed_ensemble(
        x_train, x_validation, shared, settings["seed_ensemble_hmm"]
    )
    ensemble_candidates.to_csv(diagnostics / "phase4b_ensemble_candidates.csv", index=False)
    ensemble_members.to_csv(diagnostics / "phase4b_ensemble_members.csv", index=False)

    tie_tolerance = float(shared["tie_tolerance_loglik_per_observation"])
    pca_selection = select_candidate(pca_candidate_frame, tie_tolerance)
    ensemble_selection = select_candidate(ensemble_candidates, tie_tolerance)
    run_id = _run_id(exploratory_path, feature_manifest, benchmark_manifest)
    forecasts: list[pd.DataFrame] = []
    artifact_hashes: dict[str, str] = {}
    report_hashes: dict[str, str] = {}
    for path in output.glob("*.joblib"):
        path.unlink()
    (output / "validation_forecasts.parquet").unlink(missing_ok=True)

    pca_selected: dict[str, int] | None = None
    if pca_selection is not None:
        state_count = int(pca_selection["state_count"])
        seed = int(pca_selection["best_seed"])
        model = pca_models[state_count]
        probabilities, _ = _hmm_filter(
            model, np.vstack([x_train_pca, x_validation_pca])
        )
        train_probabilities = probabilities[: len(x_train)]
        validation_probabilities = probabilities[len(x_train) :]
        state_variances = training_state_variances(
            train_probabilities, train["information_date"], train_targets
        )
        forecasts.append(
            pd.DataFrame(
                {
                    "information_date": validation["information_date"],
                    "family": "hmm_pca_exploratory",
                    "analysis_role": "primary_exploratory",
                    "state_count": state_count,
                    "member_count": 1,
                    "variance_forecast": validation_probabilities @ state_variances,
                }
            )
        )
        artifact = {
            "classification": "exploratory",
            "analysis_role": "primary_exploratory",
            "run_id": run_id,
            "pca": pca,
            "model": model,
            "feature_names": list(MODEL_FEATURES),
            "state_variances": state_variances,
            "state_count": state_count,
            "seed": seed,
        }
        risk_order = np.argsort(state_variances)
        state_labels = ["" for _ in range(state_count)]
        for rank, state in enumerate(risk_order, start=1):
            if state_count == 2:
                label = "lower_training_risk" if rank == 1 else "higher_training_risk"
            else:
                label = f"training_risk_rank_{rank}"
            state_labels[int(state)] = label
        artifact["state_labels"] = state_labels
        artifact_path = output / "hmm_pca_exploratory.joblib"
        joblib.dump(artifact, artifact_path)
        artifact_hashes[artifact_path.name] = _sha256(artifact_path)

        component_names = [f"PC{index + 1}" for index in range(pca.n_components_)]
        loadings = pd.DataFrame(
            pca.components_.T, index=list(MODEL_FEATURES), columns=component_names
        )
        loadings.index.name = "feature"
        loadings_path = diagnostics / "phase4b_pca_loadings.csv"
        loadings.reset_index().to_csv(loadings_path, index=False)
        report_hashes[loadings_path.name] = _sha256(loadings_path)

        explained = pd.DataFrame(
            {
                "component": component_names,
                "explained_variance_ratio": pca.explained_variance_ratio_,
            }
        )
        explained_path = diagnostics / "phase4b_pca_explained_variance.csv"
        explained.to_csv(explained_path, index=False)
        report_hashes[explained_path.name] = _sha256(explained_path)

        original_feature_means = pca.inverse_transform(model.means_)
        profiles = pd.DataFrame(original_feature_means, columns=list(MODEL_FEATURES))
        profiles.insert(0, "training_variance_mapping", state_variances)
        profiles.insert(0, "state_label", state_labels)
        profiles.insert(0, "state", np.arange(state_count))
        profiles_path = diagnostics / "phase4b_pca_state_profiles.csv"
        profiles.to_csv(profiles_path, index=False)
        report_hashes[profiles_path.name] = _sha256(profiles_path)

        transition = pd.DataFrame(
            model.transmat_,
            index=[f"from_{label}" for label in state_labels],
            columns=[f"to_{label}" for label in state_labels],
        )
        transition.index.name = "origin"
        transition_path = diagnostics / "phase4b_pca_transition_matrix.csv"
        transition.reset_index().to_csv(transition_path, index=False)
        report_hashes[transition_path.name] = _sha256(transition_path)
        pca_selected = {"state_count": state_count, "seed": seed}

    ensemble_selected: dict[str, Any] | None = None
    if ensemble_selection is not None:
        state_count = int(ensemble_selection["state_count"])
        members = ensemble_models[state_count]
        member_forecasts: list[np.ndarray] = []
        artifacts: list[dict[str, Any]] = []
        for model, row in members:
            probabilities, _ = _hmm_filter(model, np.vstack([x_train, x_validation]))
            state_variances = training_state_variances(
                probabilities[: len(x_train)], train["information_date"], train_targets
            )
            member_forecasts.append(probabilities[len(x_train) :] @ state_variances)
            artifacts.append(
                {
                    "model": model,
                    "seed": int(row["seed"]),
                    "state_variances": state_variances,
                }
            )
        forecasts.append(
            pd.DataFrame(
                {
                    "information_date": validation["information_date"],
                    "family": "hmm_cross_asset_seed_ensemble",
                    "analysis_role": "robustness",
                    "state_count": state_count,
                    "member_count": len(members),
                    "variance_forecast": np.mean(member_forecasts, axis=0),
                }
            )
        )
        artifact = {
            "classification": "exploratory",
            "analysis_role": "robustness",
            "run_id": run_id,
            "feature_names": list(MODEL_FEATURES),
            "state_count": state_count,
            "aggregation": "equal_weight",
            "members": artifacts,
        }
        artifact_path = output / "hmm_cross_asset_seed_ensemble.joblib"
        joblib.dump(artifact, artifact_path)
        artifact_hashes[artifact_path.name] = _sha256(artifact_path)
        ensemble_selected = {
            "state_count": state_count,
            "member_count": len(members),
            "member_seeds": [int(row["seed"]) for _, row in members],
        }

    if forecasts:
        validation_forecasts = pd.concat(forecasts, ignore_index=True)
    else:
        validation_forecasts = pd.DataFrame(
            columns=[
                "information_date",
                "family",
                "analysis_role",
                "state_count",
                "member_count",
                "variance_forecast",
            ]
        )
    if not validation_forecasts.empty:
        if not np.isfinite(validation_forecasts["variance_forecast"]).all():
            raise ValueError("Exploratory validation forecasts contain non-finite values")
        if (validation_forecasts["variance_forecast"] <= 0).any():
            raise ValueError("Exploratory variance forecasts must be positive")
    forecast_path = output / "validation_forecasts.parquet"
    validation_forecasts.to_parquet(forecast_path, index=False)
    artifact_hashes[forecast_path.name] = _sha256(forecast_path)

    result = ExploratoryBuildResult(
        run_id=run_id,
        training_rows=len(train),
        validation_rows=len(validation),
        pca_selected=pca_selected,
        ensemble_selected=ensemble_selected,
    )
    manifest = {
        **asdict(result),
        "exploratory_version": EXPLORATORY_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "exploratory_config_sha256": _sha256(exploratory_path),
        "feature_manifest_sha256": _sha256(feature_manifest),
        "benchmark_manifest_sha256": _sha256(benchmark_manifest),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "post_stabilization_roles": settings["post_stabilization_roles"],
        "artifacts": artifact_hashes,
        "reports": report_hashes,
    }
    (output / "exploratory_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return result
