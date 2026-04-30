from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


@dataclass
class Scenario:
    n_ev: int
    n_slots: int
    delta_h: float
    arrivals: np.ndarray
    departures: np.ndarray
    energy_kwh: np.ndarray
    pmax_kw: np.ndarray
    eta: np.ndarray
    price: np.ndarray
    base_forecast_kw: np.ndarray
    base_actual_kw: np.ndarray
    base_sigma_kw: np.ndarray
    transformer_capacity_kw: np.ndarray


@dataclass
class MethodResult:
    name: str
    schedule_kw: np.ndarray
    remaining_kwh: np.ndarray
    runtime_s: float
    extra: dict


def build_scenario(
    n_ev: int = 50,
    n_slots: int = 48,
    seed: int = 7,
    capacity_factor: float = 0.32,
    price_load_mode: str = "aligned",
) -> Scenario:
    rng = np.random.default_rng(seed)
    delta_h = 24.0 / n_slots
    slot_hours = np.arange(n_slots) * delta_h

    morning_peak = np.exp(-0.5 * ((slot_hours - 9.0) / 2.0) ** 2)
    evening_peak = 1.8 * np.exp(-0.5 * ((slot_hours - 18.0) / 3.0) ** 2)
    arrival_prob = 0.15 + morning_peak + evening_peak
    arrival_prob = arrival_prob / arrival_prob.sum()

    arrivals = rng.choice(np.arange(n_slots - 3), size=n_ev, p=arrival_prob[:-3] / arrival_prob[:-3].sum())
    parking_slots = rng.integers(5, 19, size=n_ev)
    departures = np.minimum(arrivals + parking_slots, n_slots)
    pmax_kw = rng.choice(np.array([6.6, 7.2, 11.0]), size=n_ev, p=np.array([0.35, 0.45, 0.20]))
    eta = rng.uniform(0.90, 0.96, size=n_ev)

    max_feasible_energy = (departures - arrivals) * delta_h * pmax_kw * eta
    requested_energy = rng.uniform(8.0, 32.0, size=n_ev)
    energy_kwh = np.minimum(requested_energy, 0.88 * max_feasible_energy)
    energy_kwh = np.maximum(energy_kwh, np.minimum(4.0, 0.5 * max_feasible_energy))

    base_forecast_kw = (
        42.0
        + 12.0 * np.sin((slot_hours - 6.0) / 24.0 * 2.0 * np.pi)
        + 28.0 * np.exp(-0.5 * ((slot_hours - 19.0) / 3.2) ** 2)
    )
    base_forecast_kw = np.clip(base_forecast_kw, 25.0, None)
    base_shape = (base_forecast_kw - base_forecast_kw.min()) / (np.ptp(base_forecast_kw) + 1e-9)
    if price_load_mode == "aligned":
        price = 0.35 + 0.65 * base_shape + 0.12 * np.exp(-0.5 * ((slot_hours - 8.0) / 1.8) ** 2)
    elif price_load_mode == "inverted":
        price = 0.35 + 0.65 * (1.0 - base_shape) + 0.08 * np.exp(-0.5 * ((slot_hours - 13.0) / 2.5) ** 2)
    elif price_load_mode == "flat":
        price = np.full(n_slots, 0.65)
    else:
        raise ValueError(f"Unknown price_load_mode: {price_load_mode}")
    price += rng.normal(0.0, 0.025, size=n_slots)
    price = np.clip(price, 0.30, None)
    base_sigma_kw = 0.06 * base_forecast_kw + 1.5
    base_actual_kw = base_forecast_kw + rng.normal(0.0, base_sigma_kw)
    base_actual_kw = np.clip(base_actual_kw, 15.0, None)

    ev_capacity_nominal = max(35.0, capacity_factor * n_ev * float(np.mean(pmax_kw)))
    transformer_capacity_kw = np.full(n_slots, float(base_forecast_kw.max() + ev_capacity_nominal))

    return Scenario(
        n_ev=n_ev,
        n_slots=n_slots,
        delta_h=delta_h,
        arrivals=arrivals,
        departures=departures,
        energy_kwh=energy_kwh,
        pmax_kw=pmax_kw,
        eta=eta,
        price=price,
        base_forecast_kw=base_forecast_kw,
        base_actual_kw=base_actual_kw,
        base_sigma_kw=base_sigma_kw,
        transformer_capacity_kw=transformer_capacity_kw,
    )


def available_capacity_kw(scenario: Scenario, kappa: float) -> np.ndarray:
    risk_buffer = kappa * scenario.base_sigma_kw
    cap = scenario.transformer_capacity_kw - scenario.base_forecast_kw - risk_buffer
    return np.maximum(cap, 0.0)


def active_mask(scenario: Scenario, t: int) -> np.ndarray:
    return (scenario.arrivals <= t) & (t < scenario.departures)


def update_remaining(
    remaining: np.ndarray,
    scenario: Scenario,
    schedule_kw: np.ndarray,
    t: int,
) -> None:
    delivered = schedule_kw[:, t] * scenario.eta * scenario.delta_h
    remaining[:] = np.maximum(remaining - delivered, 0.0)


def allocate_by_priority(
    desired_kw: np.ndarray,
    priority: np.ndarray,
    cap_kw: float,
) -> np.ndarray:
    out = np.zeros_like(desired_kw)
    if cap_kw <= 0:
        return out
    order = np.argsort(-priority)
    left = cap_kw
    for idx in order:
        if desired_kw[idx] <= 0 or left <= 1e-9:
            continue
        charge = min(float(desired_kw[idx]), left)
        out[idx] = charge
        left -= charge
    return out


def run_uncontrolled(scenario: Scenario, kappa: float = 1.0) -> MethodResult:
    start = time.perf_counter()
    cap = available_capacity_kw(scenario, kappa)
    schedule = np.zeros((scenario.n_ev, scenario.n_slots))
    remaining = scenario.energy_kwh.copy()

    for t in range(scenario.n_slots):
        active = active_mask(scenario, t) & (remaining > 1e-6)
        max_by_energy = remaining / (scenario.eta * scenario.delta_h)
        desired = np.where(active, np.minimum(scenario.pmax_kw, max_by_energy), 0.0)
        total = float(desired.sum())
        if total > cap[t] and total > 0:
            desired *= cap[t] / total
        schedule[:, t] = desired
        update_remaining(remaining, scenario, schedule, t)

    return MethodResult("uncontrolled_capped", schedule, remaining, time.perf_counter() - start, {})


def run_greedy_deadline_price(scenario: Scenario, kappa: float = 1.0) -> MethodResult:
    start = time.perf_counter()
    cap = available_capacity_kw(scenario, kappa)
    schedule = np.zeros((scenario.n_ev, scenario.n_slots))
    remaining = scenario.energy_kwh.copy()
    cheap_threshold = float(np.quantile(scenario.price, 0.40))

    for t in range(scenario.n_slots):
        active = active_mask(scenario, t) & (remaining > 1e-6)
        slots_left = np.maximum(scenario.departures - t, 1)
        max_by_energy = remaining / (scenario.eta * scenario.delta_h)
        required_rate = remaining / (scenario.eta * scenario.delta_h * slots_left)
        urgency = remaining / slots_left

        if scenario.price[t] <= cheap_threshold:
            desired = np.where(active, np.minimum(scenario.pmax_kw, max_by_energy), 0.0)
        else:
            desired = np.where(active, np.minimum(required_rate, max_by_energy), 0.0)

        desired = np.minimum(desired, scenario.pmax_kw)
        schedule[:, t] = allocate_by_priority(desired, urgency, float(cap[t]))
        update_remaining(remaining, scenario, schedule, t)

    return MethodResult("greedy_deadline_price", schedule, remaining, time.perf_counter() - start, {})


def local_cheapest_schedule(scenario: Scenario, adjusted_slot_cost: np.ndarray) -> np.ndarray:
    schedule = np.zeros((scenario.n_ev, scenario.n_slots))
    for i in range(scenario.n_ev):
        slots = np.arange(scenario.arrivals[i], scenario.departures[i])
        if slots.size == 0:
            continue
        order = slots[np.argsort(adjusted_slot_cost[slots])]
        remaining_power_sum = scenario.energy_kwh[i] / (scenario.eta[i] * scenario.delta_h)
        for t in order:
            if remaining_power_sum <= 1e-9:
                break
            charge = min(float(scenario.pmax_kw[i]), remaining_power_sum)
            schedule[i, t] = charge
            remaining_power_sum -= charge
    return schedule


def repair_capacity_and_energy(
    schedule: np.ndarray,
    scenario: Scenario,
    cap: np.ndarray,
) -> np.ndarray:
    repaired = schedule.copy()
    load = repaired.sum(axis=0)
    for t in range(scenario.n_slots):
        if load[t] > cap[t] and load[t] > 0:
            repaired[:, t] *= cap[t] / load[t]

    load = repaired.sum(axis=0)
    residual_cap = np.maximum(cap - load, 0.0)
    delivered = np.zeros(scenario.n_ev)
    for i in range(scenario.n_ev):
        delivered[i] = float(np.sum(repaired[i, :] * scenario.eta[i] * scenario.delta_h))

    for i in range(scenario.n_ev):
        missing_kwh = scenario.energy_kwh[i] - delivered[i]
        if missing_kwh <= 1e-8:
            continue
        slots = np.arange(scenario.arrivals[i], scenario.departures[i])
        order = slots[np.argsort(scenario.price[slots])]
        for t in order:
            if missing_kwh <= 1e-8:
                break
            room_ev = max(float(scenario.pmax_kw[i] - repaired[i, t]), 0.0)
            room_grid = float(residual_cap[t])
            charge = min(room_ev, room_grid, missing_kwh / (scenario.eta[i] * scenario.delta_h))
            if charge <= 0:
                continue
            repaired[i, t] += charge
            residual_cap[t] -= charge
            missing_kwh -= charge * scenario.eta[i] * scenario.delta_h
    return repaired


def run_dual_decomposition(
    scenario: Scenario,
    kappa: float = 1.0,
    iterations: int = 250,
    step0: float = 0.025,
) -> MethodResult:
    start = time.perf_counter()
    cap = available_capacity_kw(scenario, kappa)
    lambda_price = np.zeros(scenario.n_slots)
    schedule = np.zeros((scenario.n_ev, scenario.n_slots))

    for it in range(1, iterations + 1):
        adjusted_cost = scenario.price * scenario.delta_h + lambda_price
        schedule = local_cheapest_schedule(scenario, adjusted_cost)
        overload = schedule.sum(axis=0) - cap
        step = step0 / np.sqrt(it)
        lambda_price = np.maximum(lambda_price + step * overload, 0.0)

    before_delivered = np.zeros(scenario.n_ev)
    for i in range(scenario.n_ev):
        before_delivered[i] = float(np.sum(schedule[i, :] * scenario.eta[i] * scenario.delta_h))
    before_remaining = np.maximum(scenario.energy_kwh - before_delivered, 0.0)
    before_total_load = schedule.sum(axis=0) + scenario.base_actual_kw
    before_capacity_violation = before_total_load > scenario.transformer_capacity_kw + 1e-6

    schedule = repair_capacity_and_energy(schedule, scenario, cap)
    delivered = np.zeros(scenario.n_ev)
    for i in range(scenario.n_ev):
        delivered[i] = float(np.sum(schedule[i, :] * scenario.eta[i] * scenario.delta_h))
    remaining = np.maximum(scenario.energy_kwh - delivered, 0.0)
    extra = {
        "dual_iterations": iterations,
        "before_repair_unserved_energy_ratio": float(before_remaining.sum() / max(scenario.energy_kwh.sum(), 1e-9)),
        "before_repair_capacity_violation_rate": float(np.mean(before_capacity_violation)),
        "before_repair_capacity_violation_max_kw": float(
            np.maximum(before_total_load - scenario.transformer_capacity_kw, 0.0).max()
        ),
    }
    return MethodResult("dual_decomposition", schedule, remaining, time.perf_counter() - start, extra)


def project_box_capped_sum(
    v: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cap: float,
) -> np.ndarray:
    if lower.sum() > cap and lower.sum() > 0:
        lower = lower * (cap / lower.sum())
        upper = np.maximum(upper, lower)

    z = np.clip(v, lower, upper)
    if z.sum() <= cap + 1e-9:
        return z

    low = float(np.min(v - upper))
    high = float(np.max(v - lower))
    for _ in range(70):
        mid = 0.5 * (low + high)
        z = np.clip(v - mid, lower, upper)
        if z.sum() > cap:
            low = mid
        else:
            high = mid
    return np.clip(v - high, lower, upper)


def project_box_equal_sum(y: np.ndarray, upper: np.ndarray, target_sum: float) -> np.ndarray:
    target_sum = min(max(float(target_sum), 0.0), float(upper.sum()))
    if target_sum <= 1e-12:
        return np.zeros_like(y)

    low = float(np.min(y - upper))
    high = float(np.max(y))
    for _ in range(70):
        mid = 0.5 * (low + high)
        x = np.clip(y - mid, 0.0, upper)
        if x.sum() > target_sum:
            low = mid
        else:
            high = mid
    return np.clip(y - high, 0.0, upper)


def run_offline_admm(
    scenario: Scenario,
    kappa: float = 1.0,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-2,
) -> MethodResult:
    start = time.perf_counter()
    n, T = scenario.n_ev, scenario.n_slots
    cap = available_capacity_kw(scenario, kappa)
    upper = np.zeros((n, T))
    for i in range(n):
        upper[i, scenario.arrivals[i] : scenario.departures[i]] = scenario.pmax_kw[i]

    required_power_sum = scenario.energy_kwh / (scenario.eta * scenario.delta_h)
    p = np.zeros((n, T))
    z = np.zeros((n, T))
    u = np.zeros((n, T))
    slot_cost = scenario.price * scenario.delta_h

    primal = dual = np.inf
    for it in range(1, max_iter + 1):
        for i in range(n):
            y = z[i, :] - u[i, :] - slot_cost / rho
            p[i, :] = project_box_equal_sum(y, upper[i, :], required_power_sum[i])

        z_old = z.copy()
        y_all = p + u
        for t in range(T):
            z[:, t] = project_box_capped_sum(
                y_all[:, t],
                np.zeros(n),
                upper[:, t],
                float(cap[t]),
            )

        u = u + p - z
        primal = float(np.linalg.norm(p - z))
        dual = float(rho * np.linalg.norm(z - z_old))
        if primal <= tol and dual <= tol:
            break

    delivered = np.zeros(n)
    for i in range(n):
        delivered[i] = float(np.sum(z[i, :] * scenario.eta[i] * scenario.delta_h))
    remaining = np.maximum(scenario.energy_kwh - delivered, 0.0)
    extra = {
        "offline_admm_iterations": it,
        "offline_admm_primal_residual": primal,
        "offline_admm_dual_residual": dual,
    }
    return MethodResult("offline_admm", z, remaining, time.perf_counter() - start, extra)


def admm_per_slot(
    linear_cost: np.ndarray,
    pmax: np.ndarray,
    cap_kw: float,
    p_ref_kw: float,
    smooth_weight: float,
    lower: np.ndarray | None = None,
    alpha: float = 0.02,
    rho: float = 1.0,
    max_iter: int = 80,
    tol: float = 1e-3,
) -> tuple[np.ndarray, int, float, float]:
    n = len(linear_cost)
    if n == 0 or cap_kw <= 0:
        return np.zeros(n), 0, 0.0, 0.0

    if lower is None:
        lower = np.zeros(n)
    lower = np.minimum(np.maximum(lower, 0.0), pmax)
    if lower.sum() > cap_kw and lower.sum() > 0:
        lower = lower * (cap_kw / lower.sum())

    z = np.zeros(n)
    u = np.zeros(n)
    p = np.zeros(n)
    q = max(float(smooth_weight), 0.0)
    cap_kw = min(float(cap_kw), float(pmax.sum()))

    primal = dual = np.inf
    for it in range(1, max_iter + 1):
        p = (rho * (z - u) - linear_cost) / (alpha + rho)
        p = np.clip(p, lower, pmax)

        z_old = z.copy()
        v = p + u
        if q > 0:
            shift = (2.0 * q / (rho + 2.0 * q * n)) * (float(v.sum()) - float(p_ref_kw))
            z_unconstrained = v - shift
        else:
            z_unconstrained = v
        z = project_box_capped_sum(z_unconstrained, lower, pmax, cap_kw)

        u = u + p - z
        primal = float(np.linalg.norm(p - z))
        dual = float(rho * np.linalg.norm(z - z_old))
        if primal <= tol and dual <= tol:
            break

    return z, it, primal, dual


def run_online_lyapunov_admm(
    scenario: Scenario,
    kappa: float = 1.0,
    V: float = 1.0,
    beta: float = 0.015,
    urgency_delta: float = 1.0,
    max_iter: int = 80,
    use_deadline_floor: bool = True,
    use_queue_urgency: bool = True,
    name: str = "online_lyapunov_admm",
) -> MethodResult:
    start = time.perf_counter()
    cap = available_capacity_kw(scenario, kappa)
    schedule = np.zeros((scenario.n_ev, scenario.n_slots))
    remaining = scenario.energy_kwh.copy()
    total_iters = 0
    used_slots = 0
    final_residuals = []

    for t in range(scenario.n_slots):
        active_idx = np.flatnonzero(active_mask(scenario, t) & (remaining > 1e-6))
        if active_idx.size == 0:
            continue

        slots_left = np.maximum(scenario.departures[active_idx] - t, 1)
        max_by_energy = remaining[active_idx] / (scenario.eta[active_idx] * scenario.delta_h)
        pmax = np.minimum(scenario.pmax_kw[active_idx], max_by_energy)
        if use_deadline_floor:
            future_slots_after_now = np.maximum(slots_left - 1, 0)
            future_max_energy = (
                future_slots_after_now
                * scenario.pmax_kw[active_idx]
                * scenario.eta[active_idx]
                * scenario.delta_h
            )
            lower = np.maximum(
                (remaining[active_idx] - future_max_energy)
                / (scenario.eta[active_idx] * scenario.delta_h),
                0.0,
            )
            lower = np.minimum(lower, pmax)
        else:
            lower = np.zeros_like(pmax)
        urgency_weight = 1.0 / (slots_left + urgency_delta) if use_queue_urgency else np.zeros_like(slots_left, dtype=float)

        linear = (
            V * scenario.price[t] * scenario.delta_h
            - urgency_weight * remaining[active_idx] * scenario.eta[active_idx] * scenario.delta_h
        )
        p_ref = 0.82 * float(cap[t])
        smooth_weight = V * beta
        p, iters, primal, dual = admm_per_slot(
            linear,
            pmax,
            float(cap[t]),
            p_ref,
            smooth_weight,
            lower=lower,
            max_iter=max_iter,
        )
        schedule[active_idx, t] = p
        update_remaining(remaining, scenario, schedule, t)
        total_iters += iters
        used_slots += 1
        final_residuals.append((primal, dual))

    extra = {
        "total_admm_iterations": total_iters,
        "active_slots": used_slots,
        "avg_admm_iterations": total_iters / max(used_slots, 1),
        "avg_final_primal_residual": float(np.mean([x[0] for x in final_residuals])) if final_residuals else 0.0,
        "avg_final_dual_residual": float(np.mean([x[1] for x in final_residuals])) if final_residuals else 0.0,
        "V": V,
        "beta": beta,
    }
    return MethodResult(name, schedule, remaining, time.perf_counter() - start, extra)


def run_offline_centralized_lp(scenario: Scenario, kappa: float = 1.0) -> MethodResult:
    start = time.perf_counter()
    n, T = scenario.n_ev, scenario.n_slots
    cap = available_capacity_kw(scenario, kappa)
    n_power = n * T
    n_vars = n_power + n

    c = np.zeros(n_vars)
    for i in range(n):
        for t in range(T):
            c[i * T + t] = scenario.price[t] * scenario.delta_h
    c[n_power:] = 1000.0

    bounds = []
    for i in range(n):
        for t in range(T):
            if scenario.arrivals[i] <= t < scenario.departures[i]:
                bounds.append((0.0, float(scenario.pmax_kw[i])))
            else:
                bounds.append((0.0, 0.0))
    bounds.extend([(0.0, None) for _ in range(n)])

    rows = T + n
    A = lil_matrix((rows, n_vars))
    b = np.zeros(rows)

    for t in range(T):
        for i in range(n):
            A[t, i * T + t] = 1.0
        b[t] = cap[t]

    for i in range(n):
        row = T + i
        for t in range(T):
            A[row, i * T + t] = -scenario.eta[i] * scenario.delta_h
        A[row, n_power + i] = -1.0
        b[row] = -scenario.energy_kwh[i]

    res = linprog(c, A_ub=A.tocsr(), b_ub=b, bounds=bounds, method="highs")
    if not res.success:
        schedule = np.zeros((n, T))
        remaining = scenario.energy_kwh.copy()
        status = res.message
    else:
        schedule = res.x[:n_power].reshape(n, T)
        delivered = schedule.sum(axis=1) * 0.0
        for i in range(n):
            delivered[i] = float(np.sum(schedule[i, :] * scenario.eta[i] * scenario.delta_h))
        remaining = np.maximum(scenario.energy_kwh - delivered, 0.0)
        status = res.message

    return MethodResult(
        "offline_centralized_lp",
        schedule,
        remaining,
        time.perf_counter() - start,
        {"solver_status": status},
    )


def evaluate_result(scenario: Scenario, result: MethodResult) -> dict:
    ev_load = result.schedule_kw.sum(axis=0)
    total_load = ev_load + scenario.base_actual_kw
    delivered = np.zeros(scenario.n_ev)
    for i in range(scenario.n_ev):
        delivered[i] = float(np.sum(result.schedule_kw[i, :] * scenario.eta[i] * scenario.delta_h))
    satisfaction = np.divide(
        np.minimum(delivered, scenario.energy_kwh),
        scenario.energy_kwh,
        out=np.zeros_like(delivered),
        where=scenario.energy_kwh > 0,
    )
    fairness = (float(satisfaction.sum()) ** 2) / (
        scenario.n_ev * float(np.sum(satisfaction**2)) + 1e-12
    )
    duration_slots = np.maximum(scenario.departures - scenario.arrivals, 1)
    feasible_energy = duration_slots * scenario.pmax_kw * scenario.eta * scenario.delta_h
    tightness = np.divide(
        scenario.energy_kwh,
        feasible_energy,
        out=np.zeros_like(scenario.energy_kwh),
        where=feasible_energy > 0,
    )
    deadline_weights = tightness / max(float(tightness.sum()), 1e-12)
    deadline_weighted_satisfaction = float(np.sum(deadline_weights * satisfaction))

    capacity_violation = total_load > scenario.transformer_capacity_kw + 1e-6
    strict_violation = result.remaining_kwh > 1e-3
    practical_violation = result.remaining_kwh > 0.10
    severe_violation = result.remaining_kwh > 0.50

    metrics = {
        "method": result.name,
        "total_cost": float(np.sum(scenario.price * ev_load * scenario.delta_h)),
        "peak_total_load_kw": float(total_load.max()),
        "peak_ev_load_kw": float(ev_load.max()),
        "peak_to_average_ratio": float(total_load.max() / max(total_load.mean(), 1e-9)),
        "deadline_violation_rate": float(np.mean(practical_violation)),
        "strict_deadline_violation_rate": float(np.mean(strict_violation)),
        "severe_deadline_violation_rate": float(np.mean(severe_violation)),
        "average_remaining_kwh": float(np.mean(result.remaining_kwh)),
        "unserved_energy_ratio": float(result.remaining_kwh.sum() / max(scenario.energy_kwh.sum(), 1e-9)),
        "fairness_jain": float(fairness),
        "worst_user_satisfaction": float(satisfaction.min()) if satisfaction.size else 0.0,
        "p95_remaining_kwh": float(np.quantile(result.remaining_kwh, 0.95)) if result.remaining_kwh.size else 0.0,
        "deadline_weighted_satisfaction": deadline_weighted_satisfaction,
        "runtime_s": float(result.runtime_s),
        "capacity_violation_rate": float(np.mean(capacity_violation)),
        "capacity_violation_max_kw": float(np.maximum(total_load - scenario.transformer_capacity_kw, 0.0).max()),
    }
    metrics.update(result.extra)
    return metrics


def plot_load_profiles(scenario: Scenario, results: list[MethodResult], out_dir: Path) -> None:
    hours = np.arange(scenario.n_slots) * scenario.delta_h
    plt.figure(figsize=(10, 5.8))
    plt.plot(hours, scenario.transformer_capacity_kw, color="black", linewidth=1.8, label="Transformer capacity")
    plt.plot(hours, scenario.base_actual_kw, color="#6b7280", linewidth=1.8, label="Base load")
    for result in results:
        total_load = scenario.base_actual_kw + result.schedule_kw.sum(axis=0)
        plt.plot(hours, total_load, linewidth=1.5, label=result.name)
    plt.xlabel("Hour")
    plt.ylabel("Power (kW)")
    plt.title("Total load profiles under different charging policies")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "load_profiles.png", dpi=180)
    plt.close()


def plot_method_comparison(metrics: pd.DataFrame, out_dir: Path) -> None:
    plot_df = metrics.set_index("method")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    items = [
        ("total_cost", "Total cost"),
        ("peak_total_load_kw", "Peak total load"),
        ("deadline_violation_rate", "Deadline violation"),
        ("average_remaining_kwh", "Avg remaining kWh"),
        ("fairness_jain", "Jain fairness"),
        ("runtime_s", "Runtime (s)"),
    ]
    for ax, (col, title) in zip(axes.ravel(), items):
        plot_df[col].plot(kind="bar", ax=ax, color="#3b82f6")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_dir / "method_comparison.png", dpi=180)
    plt.close(fig)


def plot_capacity_sweep(summary: pd.DataFrame, out_dir: Path) -> None:
    methods = list(summary["method"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    metrics = [
        ("total_cost", "Total cost"),
        ("peak_total_load_kw", "Peak total load (kW)"),
        ("unserved_energy_ratio", "Unserved energy ratio"),
        ("capacity_violation_rate", "Capacity violation rate"),
    ]
    for ax, (col, title) in zip(axes.ravel(), metrics):
        for method in methods:
            part = summary[summary["method"] == method].sort_values("capacity_factor")
            ax.plot(part["capacity_factor"], part[col], marker="o", linewidth=1.8, label=method)
        ax.set_title(title)
        ax.set_xlabel("Capacity factor")
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "capacity_sweep_summary.png", dpi=180)
    plt.close(fig)


def plot_v_sweep(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    metrics = [
        ("total_cost", "Total cost"),
        ("peak_total_load_kw", "Peak total load (kW)"),
        ("deadline_violation_rate", "Deadline violation rate"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        ax.plot(summary["V"], summary[col], marker="o", linewidth=2.0, color="#2563eb")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Lyapunov V")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "lyapunov_v_sweep.png", dpi=180)
    plt.close(fig)


def plot_risk_sweep(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    metrics = [
        ("capacity_violation_rate", "Capacity violation rate"),
        ("total_cost", "Total cost"),
        ("peak_total_load_kw", "Peak total load (kW)"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        ax.plot(summary["kappa"], summary[col], marker="o", linewidth=2.0, color="#047857")
        ax.set_title(title)
        ax.set_xlabel("Risk buffer kappa")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "risk_buffer_sweep.png", dpi=180)
    plt.close(fig)


def plot_scalability_sweep(summary: pd.DataFrame, out_dir: Path) -> None:
    methods = list(summary["method"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    metrics = [
        ("runtime_s", "Runtime (s)"),
        ("peak_total_load_kw", "Peak total load (kW)"),
        ("total_cost", "Total cost"),
        ("capacity_violation_rate", "Capacity violation rate"),
    ]
    for ax, (col, title) in zip(axes.ravel(), metrics):
        for method in methods:
            part = summary[summary["method"] == method].sort_values("n_ev")
            ax.plot(part["n_ev"], part[col], marker="o", linewidth=1.8, label=method)
        ax.set_title(title)
        ax.set_xlabel("Number of EVs")
        ax.grid(alpha=0.25)
        if col == "runtime_s":
            ax.set_yscale("log")
    axes.ravel()[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "scalability_sweep_summary.png", dpi=180)
    plt.close(fig)


def run_base_experiment(
    n_ev: int,
    seed: int,
    out_dir: Path,
    kappa: float,
    V: float,
    capacity_factor: float = 0.32,
    price_load_mode: str = "aligned",
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario = build_scenario(
        n_ev=n_ev,
        seed=seed,
        capacity_factor=capacity_factor,
        price_load_mode=price_load_mode,
    )
    scenario_meta = {
        "n_ev": scenario.n_ev,
        "n_slots": scenario.n_slots,
        "delta_h": scenario.delta_h,
        "seed": seed,
        "kappa": kappa,
        "V": V,
        "capacity_factor": capacity_factor,
        "price_load_mode": price_load_mode,
        "total_requested_energy_kwh": float(scenario.energy_kwh.sum()),
        "mean_parking_slots": float(np.mean(scenario.departures - scenario.arrivals)),
        "mean_energy_kwh": float(np.mean(scenario.energy_kwh)),
        "mean_pmax_kw": float(np.mean(scenario.pmax_kw)),
    }
    (out_dir / "scenario_meta.json").write_text(json.dumps(scenario_meta, indent=2), encoding="utf-8")

    results = [
        run_uncontrolled(scenario, kappa=kappa),
        run_greedy_deadline_price(scenario, kappa=kappa),
        run_offline_centralized_lp(scenario, kappa=kappa),
        run_dual_decomposition(scenario, kappa=kappa),
        run_offline_admm(scenario, kappa=kappa),
        run_online_lyapunov_admm(scenario, kappa=kappa, V=V),
    ]
    metrics = pd.DataFrame([evaluate_result(scenario, r) for r in results])
    metrics.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    slot_df = pd.DataFrame(
        {
            "slot": np.arange(scenario.n_slots),
            "hour": np.arange(scenario.n_slots) * scenario.delta_h,
            "price": scenario.price,
            "base_forecast_kw": scenario.base_forecast_kw,
            "base_actual_kw": scenario.base_actual_kw,
            "transformer_capacity_kw": scenario.transformer_capacity_kw,
            "available_ev_capacity_kw": available_capacity_kw(scenario, kappa),
        }
    )
    for result in results:
        slot_df[f"{result.name}_ev_load_kw"] = result.schedule_kw.sum(axis=0)
    slot_df.to_csv(out_dir / "slot_timeseries.csv", index=False, encoding="utf-8-sig")

    plot_load_profiles(scenario, results, out_dir)
    plot_method_comparison(metrics, out_dir)
    return metrics


def run_online_only_metrics(
    n_ev: int,
    seed: int,
    kappa: float,
    V: float,
    capacity_factor: float,
    price_load_mode: str = "aligned",
    use_deadline_floor: bool = True,
    use_queue_urgency: bool = True,
    name: str = "online_lyapunov_admm",
) -> dict:
    scenario = build_scenario(
        n_ev=n_ev,
        seed=seed,
        capacity_factor=capacity_factor,
        price_load_mode=price_load_mode,
    )
    result = run_online_lyapunov_admm(
        scenario,
        kappa=kappa,
        V=V,
        use_deadline_floor=use_deadline_floor,
        use_queue_urgency=use_queue_urgency,
        name=name,
    )
    return evaluate_result(scenario, result)


def run_capacity_sweep(
    n_ev: int,
    seed: int,
    out_dir: Path,
    kappa: float,
    V: float,
    capacity_factors: list[float],
    price_load_mode: str = "aligned",
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for factor in capacity_factors:
        case_dir = out_dir / f"factor_{factor:g}"
        metrics = run_base_experiment(
            n_ev=n_ev,
            seed=seed,
            out_dir=case_dir,
            kappa=kappa,
            V=V,
            capacity_factor=factor,
            price_load_mode=price_load_mode,
        )
        metrics["capacity_factor"] = factor
        rows.append(metrics)
    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(out_dir / "capacity_sweep_summary.csv", index=False, encoding="utf-8-sig")
    plot_capacity_sweep(summary, out_dir)
    return summary


def run_v_sweep(
    n_ev: int,
    seed: int,
    out_dir: Path,
    kappa: float,
    capacity_factor: float,
    V_values: list[float],
    price_load_mode: str = "aligned",
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for value in V_values:
        row = run_online_only_metrics(
            n_ev=n_ev,
            seed=seed,
            kappa=kappa,
            V=value,
            capacity_factor=capacity_factor,
            price_load_mode=price_load_mode,
        )
        row["V"] = value
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "lyapunov_v_sweep.csv", index=False, encoding="utf-8-sig")
    plot_v_sweep(summary, out_dir)
    return summary


def run_risk_sweep(
    n_ev: int,
    seed: int,
    out_dir: Path,
    V: float,
    capacity_factor: float,
    kappa_values: list[float],
    price_load_mode: str = "aligned",
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for kappa in kappa_values:
        row = run_online_only_metrics(
            n_ev=n_ev,
            seed=seed,
            kappa=kappa,
            V=V,
            capacity_factor=capacity_factor,
            price_load_mode=price_load_mode,
        )
        row["kappa"] = kappa
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "risk_buffer_sweep.csv", index=False, encoding="utf-8-sig")
    plot_risk_sweep(summary, out_dir)
    return summary


def run_scalability_sweep(
    n_values: list[int],
    seed: int,
    out_dir: Path,
    kappa: float,
    V: float,
    capacity_factor: float,
    price_load_mode: str = "aligned",
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n_ev in n_values:
        metrics = run_base_experiment(
            n_ev=n_ev,
            seed=seed,
            out_dir=out_dir / f"n_{n_ev}",
            kappa=kappa,
            V=V,
            capacity_factor=capacity_factor,
            price_load_mode=price_load_mode,
        )
        metrics["n_ev"] = n_ev
        rows.append(metrics)
    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(out_dir / "scalability_sweep_summary.csv", index=False, encoding="utf-8-sig")
    plot_scalability_sweep(summary, out_dir)
    return summary


def summarize_replicates(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in set(group_cols + ["seed"])
    ]
    summary = df.groupby(group_cols)[numeric_cols].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def plot_multiseed_method_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    method_col = "method"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    items = [
        ("total_cost", "Total cost"),
        ("peak_total_load_kw", "Peak total load (kW)"),
        ("unserved_energy_ratio", "Unserved energy ratio"),
        ("runtime_s", "Runtime (s)"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), items):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        ax.bar(summary[method_col], summary[mean_col], yerr=summary[std_col].fillna(0.0), color="#3b82f6")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_dir / "multiseed_base_summary.png", dpi=180)
    plt.close(fig)


def run_multiseed_base(
    n_ev: int,
    seeds: list[int],
    out_dir: Path,
    kappa: float,
    V: float,
    capacity_factor: float,
    price_load_mode: str = "aligned",
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        metrics = run_base_experiment(
            n_ev=n_ev,
            seed=seed,
            out_dir=out_dir / f"seed_{seed}",
            kappa=kappa,
            V=V,
            capacity_factor=capacity_factor,
            price_load_mode=price_load_mode,
        )
        metrics["seed"] = seed
        rows.append(metrics)
    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(out_dir / "multiseed_base_raw.csv", index=False, encoding="utf-8-sig")
    summary = summarize_replicates(raw, ["method"])
    summary.to_csv(out_dir / "multiseed_base_summary.csv", index=False, encoding="utf-8-sig")
    plot_multiseed_method_summary(summary, out_dir)
    return summary


def plot_ablation_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    items = [
        ("total_cost", "Total cost"),
        ("peak_total_load_kw", "Peak total load (kW)"),
        ("unserved_energy_ratio", "Unserved energy ratio"),
        ("capacity_violation_rate", "Capacity violation rate"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), items):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        ax.bar(summary["method"], summary[mean_col], yerr=summary[std_col].fillna(0.0), color="#6366f1")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_summary.png", dpi=180)
    plt.close(fig)


def run_ablation_sweep(
    n_ev: int,
    seeds: list[int],
    out_dir: Path,
    kappa: float,
    V: float,
    capacity_factor: float,
    price_load_mode: str = "aligned",
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        scenario = build_scenario(
            n_ev=n_ev,
            seed=seed,
            capacity_factor=capacity_factor,
            price_load_mode=price_load_mode,
        )
        variants = [
            run_greedy_deadline_price(scenario, kappa=kappa),
            run_online_lyapunov_admm(scenario, kappa=kappa, V=V, name="full_online_ladmm"),
            run_online_lyapunov_admm(
                scenario,
                kappa=kappa,
                V=V,
                use_deadline_floor=False,
                name="no_deadline_floor",
            ),
            run_online_lyapunov_admm(
                scenario,
                kappa=0.0,
                V=V,
                name="no_risk_buffer",
            ),
            run_online_lyapunov_admm(
                scenario,
                kappa=kappa,
                V=V,
                use_queue_urgency=False,
                name="no_lyapunov_queue",
            ),
        ]
        for result in variants:
            row = evaluate_result(scenario, result)
            row["seed"] = seed
            rows.append(row)
    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "ablation_raw.csv", index=False, encoding="utf-8-sig")
    summary = summarize_replicates(raw, ["method"])
    summary.to_csv(out_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    plot_ablation_summary(summary, out_dir)
    return summary


def plot_rho_sweep(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.0))
    items = [
        ("total_cost", "Total cost"),
        ("average_remaining_kwh", "Avg remaining kWh"),
        ("offline_admm_primal_residual", "Primal residual"),
        ("offline_admm_dual_residual", "Dual residual"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), items):
        ax.plot(summary["rho"], summary[metric], marker="o", linewidth=2.0)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("ADMM rho")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "offline_admm_rho_sweep.png", dpi=180)
    plt.close(fig)


def run_rho_sweep(
    n_ev: int,
    seed: int,
    out_dir: Path,
    kappa: float,
    capacity_factor: float,
    rho_values: list[float],
    price_load_mode: str = "aligned",
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario = build_scenario(
        n_ev=n_ev,
        seed=seed,
        capacity_factor=capacity_factor,
        price_load_mode=price_load_mode,
    )
    rows = []
    for rho in rho_values:
        result = run_offline_admm(scenario, kappa=kappa, rho=rho)
        row = evaluate_result(scenario, result)
        row["rho"] = rho
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "offline_admm_rho_sweep.csv", index=False, encoding="utf-8-sig")
    plot_rho_sweep(summary, out_dir)
    return summary


def run_risk_correlation_sweep(
    n_ev: int,
    seed: int,
    out_dir: Path,
    V: float,
    capacity_factor: float,
    kappa_values: list[float],
    price_load_modes: list[str],
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in price_load_modes:
        summary = run_risk_sweep(
            n_ev=n_ev,
            seed=seed,
            out_dir=out_dir / mode,
            V=V,
            capacity_factor=capacity_factor,
            kappa_values=kappa_values,
            price_load_mode=mode,
        )
        summary["price_load_mode"] = mode
        rows.append(summary)
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(out_dir / "risk_correlation_sweep.csv", index=False, encoding="utf-8-sig")
    plot_risk_correlation_sweep(combined, out_dir)
    return combined


def plot_risk_correlation_sweep(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    metrics = [
        ("capacity_violation_rate", "Capacity violation rate"),
        ("total_cost", "Total cost"),
        ("peak_total_load_kw", "Peak total load (kW)"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        for mode in summary["price_load_mode"].unique():
            part = summary[summary["price_load_mode"] == mode].sort_values("kappa")
            ax.plot(part["kappa"], part[col], marker="o", linewidth=2.0, label=mode)
        ax.set_title(title)
        ax.set_xlabel("Risk buffer kappa")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "risk_correlation_sweep.png", dpi=180)
    plt.close(fig)


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EV charging scheduling experiments.")
    parser.add_argument(
        "--mode",
        choices=[
            "base",
            "capacity-sweep",
            "v-sweep",
            "risk-sweep",
            "risk-correlation-sweep",
            "scalability-sweep",
            "multiseed-base",
            "ablation",
            "rho-sweep",
        ],
        default="base",
    )
    parser.add_argument("--n-ev", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--V", type=float, default=1.0)
    parser.add_argument("--capacity-factor", type=float, default=0.32)
    parser.add_argument("--capacity-factors", type=str, default="0.40,0.32,0.26,0.22,0.20")
    parser.add_argument("--V-values", type=str, default="0.05,0.1,0.2,0.5,0.8,1,1.2,1.5,2,5")
    parser.add_argument("--kappa-values", type=str, default="0,0.5,1,1.5,2")
    parser.add_argument("--rho-values", type=str, default="0.1,0.5,1,2,5,10")
    parser.add_argument("--n-values", type=str, default="50,100,200,500")
    parser.add_argument("--seeds", type=str, default="7,11,13")
    parser.add_argument("--price-load-mode", choices=["aligned", "inverted", "flat"], default="aligned")
    parser.add_argument("--price-load-modes", type=str, default="aligned,inverted")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/base_experiment"))
    args = parser.parse_args()

    if args.mode == "capacity-sweep":
        metrics = run_capacity_sweep(
            n_ev=args.n_ev,
            seed=args.seed,
            out_dir=args.out_dir,
            kappa=args.kappa,
            V=args.V,
            capacity_factors=parse_float_list(args.capacity_factors),
            price_load_mode=args.price_load_mode,
        )
    elif args.mode == "v-sweep":
        metrics = run_v_sweep(
            n_ev=args.n_ev,
            seed=args.seed,
            out_dir=args.out_dir,
            kappa=args.kappa,
            capacity_factor=args.capacity_factor,
            V_values=parse_float_list(args.V_values),
            price_load_mode=args.price_load_mode,
        )
    elif args.mode == "risk-sweep":
        metrics = run_risk_sweep(
            n_ev=args.n_ev,
            seed=args.seed,
            out_dir=args.out_dir,
            V=args.V,
            capacity_factor=args.capacity_factor,
            kappa_values=parse_float_list(args.kappa_values),
            price_load_mode=args.price_load_mode,
        )
    elif args.mode == "risk-correlation-sweep":
        metrics = run_risk_correlation_sweep(
            n_ev=args.n_ev,
            seed=args.seed,
            out_dir=args.out_dir,
            V=args.V,
            capacity_factor=args.capacity_factor,
            kappa_values=parse_float_list(args.kappa_values),
            price_load_modes=[item.strip() for item in args.price_load_modes.split(",") if item.strip()],
        )
    elif args.mode == "scalability-sweep":
        metrics = run_scalability_sweep(
            n_values=parse_int_list(args.n_values),
            seed=args.seed,
            out_dir=args.out_dir,
            kappa=args.kappa,
            V=args.V,
            capacity_factor=args.capacity_factor,
            price_load_mode=args.price_load_mode,
        )
    elif args.mode == "multiseed-base":
        metrics = run_multiseed_base(
            n_ev=args.n_ev,
            seeds=parse_int_list(args.seeds),
            out_dir=args.out_dir,
            kappa=args.kappa,
            V=args.V,
            capacity_factor=args.capacity_factor,
            price_load_mode=args.price_load_mode,
        )
    elif args.mode == "ablation":
        metrics = run_ablation_sweep(
            n_ev=args.n_ev,
            seeds=parse_int_list(args.seeds),
            out_dir=args.out_dir,
            kappa=args.kappa,
            V=args.V,
            capacity_factor=args.capacity_factor,
            price_load_mode=args.price_load_mode,
        )
    elif args.mode == "rho-sweep":
        metrics = run_rho_sweep(
            n_ev=args.n_ev,
            seed=args.seed,
            out_dir=args.out_dir,
            kappa=args.kappa,
            capacity_factor=args.capacity_factor,
            rho_values=parse_float_list(args.rho_values),
            price_load_mode=args.price_load_mode,
        )
    else:
        metrics = run_base_experiment(
            n_ev=args.n_ev,
            seed=args.seed,
            out_dir=args.out_dir,
            kappa=args.kappa,
            V=args.V,
            capacity_factor=args.capacity_factor,
            price_load_mode=args.price_load_mode,
        )

    display_cols = [
        "method",
        "total_cost",
        "peak_total_load_kw",
        "deadline_violation_rate",
        "average_remaining_kwh",
        "fairness_jain",
        "runtime_s",
        "capacity_violation_rate",
    ]
    available_display_cols = [col for col in display_cols if col in metrics.columns]
    if available_display_cols:
        print(metrics[available_display_cols].to_string(index=False))
    else:
        print(metrics.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
