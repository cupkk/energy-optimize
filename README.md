# Energy Optimize EV Charging Experiments

This repository contains a reproducible experiment prototype for an online risk-aware distributed EV charging scheduling paper.

## Main Idea

The project compares offline EV charging optimization with an online Lyapunov-ADMM method. The proposed method models remaining EV charging demand as deadline-aware virtual queues, uses Lyapunov drift-plus-penalty for online control, and uses ADMM for per-slot distributed coordination.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Core Commands

Run one baseline comparison:

```powershell
python experiments/ev_charging_experiments.py --mode base --n-ev 50 --seed 7 --kappa 1.0 --V 0.8 --capacity-factor 0.32 --out-dir outputs/base_experiment
```

Run 10-seed baseline statistics:

```powershell
python experiments/ev_charging_experiments.py --mode multiseed-base --n-ev 50 --seeds 1,2,3,4,5,6,7,8,9,10 --kappa 1.0 --V 0.8 --capacity-factor 0.32 --out-dir outputs/multiseed_base_10
```

Run capacity pressure sweep:

```powershell
python experiments/ev_charging_experiments.py --mode capacity-sweep --n-ev 50 --seed 7 --kappa 1.0 --V 0.8 --capacity-factors 0.40,0.32,0.26,0.22,0.20 --out-dir outputs/capacity_sweep_v08
```

Run Lyapunov parameter sweep:

```powershell
python experiments/ev_charging_experiments.py --mode v-sweep --n-ev 50 --seed 7 --kappa 1.0 --capacity-factor 0.22 --V-values 0.05,0.1,0.2,0.5,0.8,1,1.2,1.5,2,5 --out-dir outputs/tight_v_sweep_refined
```

Run risk-buffer sweep under different price-load correlations:

```powershell
python experiments/ev_charging_experiments.py --mode risk-correlation-sweep --n-ev 50 --seed 7 --V 0.2 --capacity-factor 0.22 --kappa-values 0,0.5,1,1.5,2 --price-load-modes aligned,inverted --out-dir outputs/risk_correlation
```

Run ablation study:

```powershell
python experiments/ev_charging_experiments.py --mode ablation --n-ev 50 --seeds 7,11,13 --kappa 1.0 --V 0.8 --capacity-factor 0.22 --out-dir outputs/ablation_3_v08
```

Run offline ADMM rho sweep:

```powershell
python experiments/ev_charging_experiments.py --mode rho-sweep --n-ev 50 --seed 7 --kappa 1.0 --capacity-factor 0.32 --rho-values 0.1,0.5,1,2,5,10 --out-dir outputs/offline_admm_rho_sweep
```

## Key Output Files

Each command writes CSV summaries and PNG figures under the selected `outputs/` subdirectory. The most useful summary files are:

- `multiseed_base_summary.csv`
- `capacity_sweep_summary.csv`
- `lyapunov_v_sweep.csv`
- `risk_correlation_sweep.csv`
- `ablation_summary.csv`
- `offline_admm_rho_sweep.csv`

## Paper Draft

The current English LaTeX draft is under `paper/main.tex`, with references in `paper/references.bib`. The local environment used during development did not provide `pdflatex` or `bibtex`, so compile the PDF in a LaTeX-enabled environment.
