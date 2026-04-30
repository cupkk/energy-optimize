from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results" / "processed"


METHOD_LABELS = {
    "offline_centralized_lp": "Offline LP",
    "offline_admm": "Offline ADMM",
    "dual_decomposition": "Dual decomp.",
    "greedy_deadline_price": "Greedy",
    "uncontrolled_capped": "Immediate capped",
    "online_lyapunov_admm": "Online L-ADMM",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.35,
            "lines.linewidth": 1.4,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def figure_dir() -> Path:
    fig_dir = ROOT / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def save_figure(fig: plt.Figure, name: str) -> None:
    fig_dir = figure_dir()
    fig.savefig(fig_dir / f"{name}.pdf", facecolor="white", transparent=False)
    fig.savefig(fig_dir / f"{name}.png", dpi=300, facecolor="white", transparent=False)
    plt.close(fig)


def read_paper_csv(processed_name: str, output_relative: str) -> pd.DataFrame:
    processed_path = PROCESSED / processed_name
    if processed_path.exists():
        return pd.read_csv(processed_path)

    output_path = ROOT / output_relative
    if output_path.exists():
        return pd.read_csv(output_path)

    raise FileNotFoundError(
        "Missing paper figure data. Expected either "
        f"{processed_path} or {output_path}. Run the paper experiments or restore results/processed."
    )


def finish_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="both", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_framework_diagram() -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(7.1, 2.55))
    ax.axis("off")

    boxes = [
        ("Active EVs\n$(Q_i,d_i,p_i^{\\max})$", 0.04, 0.58, 0.16, 0.22, "#e8eef5"),
        ("Queue and\nurgency update", 0.25, 0.58, 0.16, 0.22, "#eef3e8"),
        ("EV local\nADMM update", 0.46, 0.58, 0.16, 0.22, "#f4efe8"),
        ("Aggregator\nprojection", 0.67, 0.58, 0.16, 0.22, "#f2e8ec"),
        ("Charging action\nand state update", 0.46, 0.18, 0.16, 0.22, "#e9f2f2"),
    ]
    for text, x, y, w, h, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#444444", linewidth=0.7)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    arrows = [
        ((0.20, 0.69), (0.25, 0.69)),
        ((0.41, 0.69), (0.46, 0.69)),
        ((0.62, 0.69), (0.67, 0.69)),
        ((0.75, 0.58), (0.56, 0.40)),
        ((0.46, 0.29), (0.20, 0.58)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#333333"})

    ax.text(
        0.75,
        0.42,
        "Risk-buffered capacity:\n$\\hat B_t+\\kappa\\sigma_t+\\sum_i p_i(t)\\leq C_t$",
        ha="center",
        va="center",
        fontsize=7.4,
        color="#333333",
    )
    ax.text(
        0.50,
        0.93,
        "Per-slot distributed coordination inside an online rolling horizon",
        ha="center",
        va="center",
        fontsize=8.5,
    )
    save_figure(fig, "framework_diagram")


def build_summary_figure() -> None:
    configure_style()
    baseline = read_paper_csv("multiseed_base_summary.csv", "outputs/multiseed_base_30/multiseed_base_summary.csv")
    v_sweep = read_paper_csv("lyapunov_v_sweep.csv", "outputs/tight_v_sweep_refined/lyapunov_v_sweep.csv")

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.45))

    ax = axes[0]
    baseline = baseline.sort_values("total_cost_mean")
    for _, row in baseline.iterrows():
        method = row["method"]
        if method == "online_lyapunov_admm":
            color, marker, size = "#1f4e79", "o", 36
        elif method == "offline_centralized_lp":
            color, marker, size = "#222222", "D", 28
        else:
            color, marker, size = "#7a7a7a", "s", 24
        ax.scatter(row["total_cost_mean"], row["peak_total_load_kw_mean"], s=size, color=color, marker=marker, zorder=3)
        if method in {"offline_centralized_lp", "online_lyapunov_admm"}:
            offset = (5, -10) if method == "offline_centralized_lp" else (5, 5)
            ax.annotate(
                METHOD_LABELS[method],
                (row["total_cost_mean"], row["peak_total_load_kw_mean"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=6.7,
                ha="left",
                va="top" if method == "offline_centralized_lp" else "bottom",
            )
    ax.set_xlim(baseline["total_cost_mean"].min() - 3, baseline["total_cost_mean"].max() + 10)
    ax.set_ylim(baseline["peak_total_load_kw_mean"].min() - 2, baseline["peak_total_load_kw_mean"].max() + 3)
    ax.set_title("(a) Cost-peak trade-off")
    ax.set_xlabel("Mean charging cost")
    ax.set_ylabel("Mean peak load (kW)")
    ax.grid(True, axis="both", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.02,
        0.03,
        "Other baselines shown in gray",
        transform=ax.transAxes,
        fontsize=6.5,
        color="#555555",
    )

    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(v_sweep["V"], v_sweep["total_cost"], color="#1f4e79", marker="o", markersize=3.5, label="Cost")
    ax2.plot(
        v_sweep["V"],
        v_sweep["deadline_violation_rate"],
        color="#b23a48",
        marker="s",
        markersize=3.5,
        label="Deadline violation",
    )
    ax.set_xscale("log")
    ax.set_title("(b) Lyapunov parameter sensitivity")
    ax.set_xlabel("Lyapunov parameter V")
    ax.set_ylabel("Total cost", color="#1f4e79")
    ax2.set_ylabel("Deadline violation rate", color="#b23a48")
    ax.tick_params(axis="y", colors="#1f4e79")
    ax2.tick_params(axis="y", colors="#b23a48")
    ax.grid(True, axis="both", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    handles = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in handles]
    ax.legend(handles, labels, loc="upper right", frameon=False)

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "summary_tradeoffs")


def build_capacity_pressure_figure() -> None:
    configure_style()
    df = read_paper_csv("capacity_sweep_summary.csv", "outputs/capacity_sweep_v08/capacity_sweep_summary.csv")
    methods = ["offline_centralized_lp", "greedy_deadline_price", "dual_decomposition", "online_lyapunov_admm"]
    colors = {
        "offline_centralized_lp": "#222222",
        "greedy_deadline_price": "#777777",
        "dual_decomposition": "#8c6d31",
        "online_lyapunov_admm": "#1f4e79",
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.45), sharex=True)
    for method in methods:
        sub = df[df["method"] == method].sort_values("capacity_factor")
        label = METHOD_LABELS.get(method, method)
        axes[0].plot(sub["capacity_factor"], sub["peak_total_load_kw"], marker="o", markersize=3.2, color=colors[method], label=label)
        axes[1].plot(sub["capacity_factor"], sub["unserved_energy_ratio"], marker="o", markersize=3.2, color=colors[method], label=label)
    axes[0].set_title("(a) Peak load")
    axes[0].set_ylabel("Peak load (kW)")
    axes[1].set_title("(b) Unserved energy")
    axes[1].set_ylabel("Unserved ratio")
    for ax in axes:
        ax.set_xlabel("Capacity factor")
        ax.invert_xaxis()
        finish_axes(ax)
    axes[1].legend(loc="upper left", fontsize=6.6)
    fig.tight_layout(w_pad=2.2)
    save_figure(fig, "capacity_pressure")


def build_risk_buffer_figure() -> None:
    configure_style()
    df = read_paper_csv("risk_correlation_sweep.csv", "outputs/risk_correlation/risk_correlation_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.45), sharex=True)
    legend_handles = None
    for ax, mode in zip(axes, ["aligned", "inverted"]):
        sub = df[df["price_load_mode"] == mode].sort_values("kappa")
        ax2 = ax.twinx()
        line1 = ax.plot(sub["kappa"], sub["capacity_violation_rate"], color="#b23a48", marker="s", markersize=3.4, label="Cap. violation")[0]
        line2 = ax2.plot(sub["kappa"], sub["total_cost"], color="#1f4e79", marker="o", markersize=3.4, label="Cost")[0]
        if legend_handles is None:
            legend_handles = [line1, line2]
        ax.set_title(f"({chr(97 + list(['aligned', 'inverted']).index(mode))}) {mode} price-load")
        ax.set_xlabel("Risk buffer $\\kappa$")
        ax.set_ylabel("Capacity violation", color="#b23a48")
        ax2.set_ylabel("Total cost", color="#1f4e79")
        ax.tick_params(axis="y", colors="#b23a48")
        ax2.tick_params(axis="y", colors="#1f4e79")
        finish_axes(ax)
        ax2.spines["top"].set_visible(False)
    axes[0].legend(legend_handles, ["Cap. violation", "Cost"], loc="upper right", fontsize=6.6)
    fig.tight_layout(w_pad=2.2)
    save_figure(fig, "risk_buffer")


def build_ablation_figure() -> None:
    configure_style()
    df = read_paper_csv("ablation_summary.csv", "outputs/ablation_3_v08/ablation_summary.csv")
    order = [
        "full_online_ladmm",
        "no_deadline_floor",
        "no_lyapunov_queue",
        "no_risk_buffer",
        "online_centralized_slot",
        "greedy_deadline_price",
    ]
    labels = ["Full", "No floor", "No queue", "No buffer", "Centralized", "Greedy"]
    sub = df.set_index("method").loc[order].reset_index()
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    ax2 = ax.twinx()
    bars = ax.bar(x - 0.17, sub["unserved_energy_ratio_mean"], width=0.34, color="#b23a48", alpha=0.82, label="Unserved")
    ax2.bar(x + 0.17, sub["runtime_s_mean"], width=0.34, color="#1f4e79", alpha=0.82, label="Runtime")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Unserved ratio", color="#b23a48")
    ax2.set_ylabel("Runtime (s)", color="#1f4e79")
    ax.tick_params(axis="y", colors="#b23a48")
    ax2.tick_params(axis="y", colors="#1f4e79")
    ax.set_title("Ablation under tight capacity")
    finish_axes(ax)
    ax2.spines["top"].set_visible(False)
    ax.legend([bars, ax2.containers[0]], ["Unserved", "Runtime"], loc="upper left", fontsize=6.8)
    fig.tight_layout()
    save_figure(fig, "ablation")


def build_scalability_figure() -> None:
    configure_style()
    df = read_paper_csv(
        "scalability_sweep_summary.csv",
        "outputs/scalability_sweep_with_baselines/scalability_sweep_summary.csv",
    )
    large = read_paper_csv("scalability_fast_summary.csv", "outputs/scalability_fast_large/scalability_fast_summary.csv")
    methods = ["offline_centralized_lp", "offline_admm", "dual_decomposition", "online_lyapunov_admm"]
    colors = {
        "offline_centralized_lp": "#222222",
        "offline_admm": "#7a7a7a",
        "dual_decomposition": "#8c6d31",
        "online_lyapunov_admm": "#1f4e79",
    }
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.55))
    for method in methods:
        sub = df[df["method"] == method].sort_values("n_ev")
        axes[0].plot(sub["n_ev"], sub["runtime_s"], marker="o", markersize=3.2, color=colors[method], label=METHOD_LABELS.get(method, method))
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Six-method sweep")
    axes[0].set_xlabel("Number of EVs")
    axes[0].set_ylabel("Runtime (s, log)")
    finish_axes(axes[0])
    axes[0].legend(loc="upper left", fontsize=6.2)

    for method in ["offline_centralized_lp", "online_lyapunov_admm", "greedy_deadline_price"]:
        sub = large[large["method"] == method].sort_values("n_ev")
        axes[1].plot(sub["n_ev"], sub["runtime_s"], marker="o", markersize=3.2, label=METHOD_LABELS.get(method, method))
    axes[1].set_yscale("log")
    axes[1].set_title("(b) Fast large-scale check")
    axes[1].set_xlabel("Number of EVs")
    axes[1].set_ylabel("Runtime (s, log)")
    finish_axes(axes[1])
    axes[1].legend(loc="upper left", fontsize=6.2)
    fig.tight_layout(h_pad=1.3)
    save_figure(fig, "scalability_runtime")


def build_public_data_figure() -> None:
    configure_style()
    df = read_paper_csv("real_data_multiday_summary.csv", "outputs/real_elaadnl_multiday_5/real_data_multiday_summary.csv")
    order = ["offline_centralized_lp", "offline_admm", "greedy_deadline_price", "online_lyapunov_admm", "uncontrolled_capped"]
    labels = [METHOD_LABELS.get(m, m) for m in order]
    sub = df.set_index("method").loc[order].reset_index()
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    width = 0.36
    ax.bar(x - width / 2, sub["capacity_violation_rate_mean"], width=width, color="#b23a48", alpha=0.82, label="Cap. viol.")
    ax.bar(x + width / 2, sub["deadline_violation_rate_mean"], width=width, color="#7a7a7a", alpha=0.82, label="Deadline")
    ax.set_title("Public-session violation rates")
    ax.set_ylabel("Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.legend(loc="upper left", fontsize=6.7)
    finish_axes(ax)
    fig.tight_layout()
    save_figure(fig, "public_data_check")


def build_all_figures() -> None:
    build_framework_diagram()
    build_summary_figure()
    build_capacity_pressure_figure()
    build_risk_buffer_figure()
    build_ablation_figure()
    build_scalability_figure()
    build_public_data_figure()


if __name__ == "__main__":
    build_all_figures()
