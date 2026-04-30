from __future__ import annotations

import numpy as np

from experiments.ev_charging_experiments import available_capacity_kw, build_scenario


def test_synthetic_scenario_has_valid_time_windows() -> None:
    scenario = build_scenario(n_ev=20, n_slots=48, seed=3)

    assert np.all(scenario.arrivals < scenario.departures)
    assert np.all(scenario.departures <= scenario.n_slots)
    assert np.all(scenario.energy_kwh > 0)


def test_available_capacity_is_nonnegative() -> None:
    scenario = build_scenario(n_ev=20, n_slots=48, seed=5, capacity_factor=0.22)
    cap = available_capacity_kw(scenario, kappa=1.0)

    assert cap.shape == (scenario.n_slots,)
    assert np.all(cap >= 0.0)
