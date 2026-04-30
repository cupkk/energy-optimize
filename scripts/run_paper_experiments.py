from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "ev_charging_experiments.py"
REAL_DATA_CSV = ROOT / "data" / "raw" / "elaadnl_office_parking_v2" / "202410DatasetEVOfficeParking_v0.csv"
SEEDS_30 = ",".join(str(i) for i in range(1, 31))


def run(args: list[str]) -> None:
    command = [sys.executable, str(RUNNER), *args]
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run(
        [
            "--mode",
            "multiseed-base",
            "--n-ev",
            "50",
            "--seeds",
            SEEDS_30,
            "--kappa",
            "1.0",
            "--V",
            "0.8",
            "--capacity-factor",
            "0.32",
            "--out-dir",
            "outputs/multiseed_base_30",
        ]
    )
    run(
        [
            "--mode",
            "capacity-sweep",
            "--n-ev",
            "50",
            "--seed",
            "7",
            "--kappa",
            "1.0",
            "--V",
            "0.8",
            "--capacity-factors",
            "0.40,0.32,0.26,0.22,0.20",
            "--out-dir",
            "outputs/capacity_sweep_v08",
        ]
    )
    run(
        [
            "--mode",
            "v-sweep",
            "--n-ev",
            "50",
            "--seed",
            "7",
            "--kappa",
            "1.0",
            "--capacity-factor",
            "0.22",
            "--V-values",
            "0.05,0.1,0.2,0.5,0.8,1,1.2,1.5,2,5",
            "--out-dir",
            "outputs/tight_v_sweep_refined",
        ]
    )
    run(
        [
            "--mode",
            "p-ref-sweep",
            "--n-ev",
            "50",
            "--seeds",
            SEEDS_30,
            "--kappa",
            "1.0",
            "--V",
            "0.8",
            "--capacity-factor",
            "0.22",
            "--p-ref-ratios",
            "0.7,0.8,0.82,0.9,1.0",
            "--out-dir",
            "outputs/p_ref_sweep_v08_30",
        ]
    )
    run(
        [
            "--mode",
            "risk-correlation-sweep",
            "--n-ev",
            "50",
            "--seed",
            "7",
            "--V",
            "0.2",
            "--capacity-factor",
            "0.22",
            "--kappa-values",
            "0,0.5,1,1.5,2",
            "--price-load-modes",
            "aligned,inverted",
            "--out-dir",
            "outputs/risk_correlation",
        ]
    )
    run(
        [
            "--mode",
            "ablation",
            "--n-ev",
            "50",
            "--seeds",
            "7,11,13",
            "--kappa",
            "1.0",
            "--V",
            "0.8",
            "--capacity-factor",
            "0.22",
            "--out-dir",
            "outputs/ablation_3_v08",
        ]
    )
    run(
        [
            "--mode",
            "scalability-fast",
            "--n-values",
            "1000,2000",
            "--seed",
            "7",
            "--kappa",
            "1.0",
            "--V",
            "0.8",
            "--capacity-factor",
            "0.32",
            "--method-set",
            "fast",
            "--out-dir",
            "outputs/scalability_fast_large",
        ]
    )

    if REAL_DATA_CSV.exists():
        run(
            [
                "--mode",
                "real-data-multiday",
                "--n-ev",
                "50",
                "--seed",
                "7",
                "--kappa",
                "1.0",
                "--V",
                "0.8",
                "--capacity-factor",
                "0.32",
                "--out-dir",
                "outputs/real_elaadnl_multiday_5",
            ]
        )
    else:
        print(f"Skipping real-data experiment because {REAL_DATA_CSV} was not found.", flush=True)


if __name__ == "__main__":
    main()
