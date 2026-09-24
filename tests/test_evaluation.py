import numpy as np
import pytest

from eurostress.evaluation import (
    circular_block_indices,
    evaluate_locked_test,
    paired_effect_summary,
    qlike_loss,
)


def test_qlike_is_zero_for_perfect_forecast_and_positive_otherwise() -> None:
    loss, floor_count = qlike_loss(np.array([1.0, 2.0]), np.array([1.0, 1.0]))

    assert np.isclose(loss[0], 0.0)
    assert np.isclose(loss[1], 2.0 - np.log(2.0) - 1.0)
    assert floor_count == 0


def test_qlike_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        qlike_loss(np.array([1.0]), np.array([0.0]))


def test_circular_block_bootstrap_is_deterministic_and_in_range() -> None:
    first = circular_block_indices(10, block_length=4, replications=5, seed=42)
    second = circular_block_indices(10, block_length=4, replications=5, seed=42)

    assert np.array_equal(first, second)
    assert first.shape == (5, 10)
    assert first.min() >= 0 and first.max() < 10


def test_positive_paired_effect_favors_candidate() -> None:
    candidate = np.array([1.0, 1.5, 2.0])
    reference = np.array([2.0, 2.5, 3.0])
    indices = np.tile(np.arange(3), (10, 1))

    result = paired_effect_summary(candidate, reference, indices, 0.95)

    assert result["mean_effect"] == 1.0
    assert result["ci_lower"] == 1.0
    assert result["ci_upper"] == 1.0


def test_locked_evaluator_refuses_to_run_without_explicit_unlock(tmp_path) -> None:
    with pytest.raises(PermissionError, match="remains sealed"):
        evaluate_locked_test(
            "config/evaluation_lock.yml",
            "protocol/final_evaluation_lock.json",
            tmp_path,
        )
