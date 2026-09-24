import numpy as np

from eurostress.exploratory import elite_training_members, ensemble_log_density


def test_ensemble_log_density_is_equal_weight_mixture() -> None:
    member_log_densities = np.log(np.array([[0.2, 0.8], [0.4, 0.2]]))

    result = ensemble_log_density(member_log_densities)

    assert np.allclose(np.exp(result), [0.3, 0.5])


def test_elite_member_selection_uses_training_likelihood_only() -> None:
    rows = [
        {"seed": 0, "converged": True, "training_loglik": -100.0},
        {"seed": 1, "converged": True, "training_loglik": -103.0},
        {"seed": 2, "converged": True, "training_loglik": -107.0},
        {"seed": 3, "converged": False, "training_loglik": -99.0},
    ]

    selected = elite_training_members(rows, training_rows=100, tolerance_per_observation=0.05)

    assert selected == {0, 1}
