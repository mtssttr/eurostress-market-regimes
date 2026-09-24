import numpy as np
import pandas as pd

from eurostress.models import (
    forward_filter,
    select_candidate,
    state_diagnostics,
    training_state_variances,
)


def test_forward_filter_is_causal_and_normalized() -> None:
    observations = np.array([[0.0], [0.2], [3.0]])
    start = np.array([0.8, 0.2])
    transition = np.array([[0.9, 0.1], [0.2, 0.8]])
    means = np.array([[0.0], [3.0]])
    covariances = np.array([[1.0], [1.0]])

    full_probabilities, full_density = forward_filter(
        observations, start, transition, means, covariances
    )
    prefix_probabilities, prefix_density = forward_filter(
        observations[:2], start, transition, means, covariances
    )

    assert np.allclose(full_probabilities[:2], prefix_probabilities)
    assert np.allclose(full_density[:2], prefix_density)
    assert np.allclose(full_probabilities.sum(axis=1), 1.0)


def test_state_diagnostics_use_filtered_occupancy_and_hard_runs() -> None:
    probabilities = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8], [0.1, 0.9]]
    )

    occupancy, durations = state_diagnostics(probabilities)

    assert np.allclose(occupancy, [0.42, 0.58])
    assert durations == [2.0, 3.0]


def test_state_variance_mapping_uses_aligned_training_targets() -> None:
    probabilities = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    dates = pd.Series(pd.to_datetime(["2020-01-03", "2020-01-10", "2020-01-17"]))
    targets = pd.DataFrame(
        {
            "information_date": dates,
            "realised_variance": [1.0, 3.0, 5.0],
        }
    )

    mapped = training_state_variances(probabilities, dates, targets)

    assert np.allclose(mapped, [5 / 3, 13 / 3])


def test_candidate_selection_prefers_fewer_states_within_tolerance() -> None:
    candidates = pd.DataFrame(
        {
            "state_count": [2, 3, 4],
            "validation_loglik_per_observation": [-2.005, -2.0, -2.2],
            "admissible": [True, True, True],
        }
    )

    selected = select_candidate(candidates, tie_tolerance_per_observation=0.01)

    assert selected is not None
    assert selected["state_count"] == 2


def test_candidate_selection_returns_none_if_all_fail() -> None:
    candidates = pd.DataFrame(
        {
            "state_count": [2, 3],
            "validation_loglik_per_observation": [-2.0, -1.9],
            "admissible": [False, False],
        }
    )

    assert select_candidate(candidates, 0.01) is None
