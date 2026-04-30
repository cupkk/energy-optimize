from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


METHOD_LABELS = {
    "offline_centralized_lp": "Offline LP",
    "offline_admm": "Offline ADMM",
    "dual_decomposition": "Dual decomp.",
    "greedy_deadline_price": "Greedy",
    "uncontrolled_capped": "Uncontrolled",
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
            "savefig.bbox": "tight",
        }
    )


def build_summary_figure() -> None:
    configure_style()
    fig_dir = ROOT / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(ROOT / "outputs" / "multiseed_base_30" / "multiseed_base_summary.csv")
    v_sweep = pd.read_csv(ROOT / "outputs" / "tight_v_sweep_refined" / "lyapunov_v_sweep.csv")

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
    fig.savefig(fig_dir / "summary_tradeoffs.pdf", facecolor="white", transparent=False)
    fig.savefig(fig_dir / "summary_tradeoffs.png", dpi=300, facecolor="white", transparent=False)


if __name__ == "__main__":
    build_summary_figure()
