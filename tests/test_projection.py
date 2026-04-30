from __future__ import annotations

import numpy as np

from experiments.ev_charging_experiments import project_box_capped_sum


def test_project_box_capped_sum_respects_bounds_and_cap() -> None:
    v = np.array([5.0, 1.0, 3.0])
    lower = np.array([1.0, 0.0, 0.5])
    upper = np.array([4.0, 4.0, 4.0])

    z = project_box_capped_sum(v, lower, upper, cap=5.0)

    assert np.all(z >= lower - 1e-8)
    assert np.all(z <= upper + 1e-8)
    assert z.sum() <= 5.0 + 1e-8


def test_project_box_capped_sum_relaxes_infeasible_lower_bound() -> None:
    v = np.array([4.0, 4.0])
    lower = np.array([3.0, 3.0])
    upper = np.array([5.0, 5.0])

    z = project_box_capped_sum(v, lower, upper, cap=4.0)

    assert np.all(z >= np.array([2.0, 2.0]) - 1e-8)
    assert np.all(z <= upper + 1e-8)
    assert z.sum() <= 4.0 + 1e-8
