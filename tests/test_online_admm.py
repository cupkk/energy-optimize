from __future__ import annotations

import math

import numpy as np

from experiments.ev_charging_experiments import (
    admm_per_slot,
    build_scenario,
    evaluate_result,
    run_online_lyapunov_admm,
)


def test_admm_per_slot_respects_bounds_and_capacity() -> None:
    linear = np.array([-0.8, 0.1, -0.2])
    pmax = np.array([7.2, 7.2, 11.0])
    lower = np.array([1.0, 0.0, 0.5])

    p, _, primal, dual = admm_per_slot(
        linear_cost=linear,
        pmax=pmax,
        cap_kw=8.0,
        p_ref_kw=6.4,
        smooth_weight=0.012,
        lower=lower,
    )

    assert np.all(np.isfinite(p))
    assert np.all(p >= lower - 1e-6)
    assert np.all(p <= pmax + 1e-6)
    assert p.sum() <= 8.0 + 1e-6
    assert math.isfinite(primal)
    assert math.isfinite(dual)


def test_online_lyapunov_admm_small_case_is_finite() -> None:
    scenario = build_scenario(n_ev=8, n_slots=24, seed=9, capacity_factor=0.32)
    result = run_online_lyapunov_admm(scenario, kappa=1.0, V=0.8, max_iter=40)
    metrics = evaluate_result(scenario, result)

    assert np.all(np.isfinite(result.schedule_kw))
    assert np.all(result.schedule_kw >= -1e-8)
    assert math.isfinite(metrics["total_cost"])
    assert math.isfinite(metrics["unserved_energy_ratio"])
