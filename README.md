# Energy Optimize EV Charging Experiments

This repository contains a reproducible experiment prototype for an online risk-aware distributed EV charging scheduling paper.

## Main Idea

The project compares offline EV charging optimization with an online Lyapunov-ADMM method. The proposed method models remaining EV charging demand as deadline-aware virtual queues, uses Lyapunov drift-plus-penalty for online control, and uses ADMM for per-slot distributed coordination.

## Install

```powershell
python -m pip install -r requirements.txt
```

The reported results were generated with Python 3.12.3 and the pinned packages in `requirements-lock.txt`:

```powershell
python -m pip install -r requirements-lock.txt
```

## Core Commands

Run one baseline comparison:

```powershell
python experiments/ev_charging_experiments.py --mode base --n-ev 50 --seed 7 --kappa 1.0 --V 0.8 --capacity-factor 0.32 --out-dir outputs/base_experiment
```

Run 30-seed baseline statistics:

```powershell
python experiments/ev_charging_experiments.py --mode multiseed-base --n-ev 50 --seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 --kappa 1.0 --V 0.8 --capacity-factor 0.32 --out-dir outputs/multiseed_base_30
```

Run a workplace-style semi-realistic sensitivity check:

```powershell
python experiments/ev_charging_experiments.py --mode multiseed-base --n-ev 50 --seeds 1,2,3,4,5,6,7,8,9,10 --kappa 1.0 --V 0.8 --capacity-factor 0.32 --scenario-profile workplace --out-dir outputs/workplace_multiseed_10
```

Run capacity pressure sweep:

```powershell
python experiments/ev_charging_experiments.py --mode capacity-sweep --n-ev 50 --seed 7 --kappa 1.0 --V 0.8 --capacity-factors 0.40,0.32,0.26,0.22,0.20 --out-dir outputs/capacity_sweep_v08
```

Run Lyapunov parameter sweep:

```powershell
python experiments/ev_charging_experiments.py --mode v-sweep --n-ev 50 --seed 7 --kappa 1.0 --capacity-factor 0.22 --V-values 0.05,0.1,0.2,0.5,0.8,1,1.2,1.5,2,5 --out-dir outputs/tight_v_sweep_refined
```

Run reference-margin sensitivity for `P_ref(t)=gamma(C_t-Bhat_t-kappa sigma_t)^+`:

```powershell
python experiments/ev_charging_experiments.py --mode p-ref-sweep --n-ev 50 --seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 --kappa 1.0 --V 0.8 --capacity-factor 0.22 --p-ref-ratios 0.7,0.8,0.82,0.9,1.0 --out-dir outputs/p_ref_sweep_v08_30
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

Run large-scale fast scalability check:

```powershell
python experiments/ev_charging_experiments.py --mode scalability-fast --n-values 1000,2000 --seed 7 --kappa 1.0 --V 0.8 --capacity-factor 0.32 --method-set fast --out-dir outputs/scalability_fast_large
```

Download the public 4TU/ElaadNL office parking dataset and run a real-session check:

```powershell
New-Item -ItemType Directory -Force -Path data/raw | Out-Null
Invoke-WebRequest -Uri "https://data.4tu.nl/ndownloader/items/80ef3824-3f5d-4e45-8794-3b8791efbd13/versions/2" -OutFile data/raw/elaadnl_office_parking_v2.zip -UseBasicParsing
Expand-Archive -LiteralPath data/raw/elaadnl_office_parking_v2.zip -DestinationPath data/raw/elaadnl_office_parking_v2 -Force
python experiments/ev_charging_experiments.py --mode real-data-multiday --n-ev 50 --seed 7 --kappa 1.0 --V 0.8 --capacity-factor 0.32 --out-dir outputs/real_elaadnl_multiday_5
```

Run the main paper experiment set:

```powershell
python scripts/run_paper_experiments.py
```

Run tests:

```powershell
python -m pytest
```

## Key Output Files

Each command writes CSV summaries and PNG figures under the selected `outputs/` subdirectory. The most useful summary files are:

- `multiseed_base_summary.csv`
- `capacity_sweep_summary.csv`
- `lyapunov_v_sweep.csv`
- `p_ref_sweep_summary.csv`
- `risk_correlation_sweep.csv`
- `ablation_summary.csv`
- `offline_admm_rho_sweep.csv`
- `scalability_fast_summary.csv`
- `real_data_multiday_summary.csv`

The lightweight CSV files used directly by the paper figure script are tracked under `results/processed/`. The larger `outputs/` directory is still ignored for new generated files.

## Scenario Profiles

- `synthetic`: default mixed daily-arrival profile with morning and evening peaks.
- `workplace`: semi-realistic workplace profile with morning arrivals, workday-length parking, and lognormal charging demand. This is a sensitivity setting, not a replacement for validation on raw public charging data.
- `real_csv`: public-session profile created by `--mode real-data-base` or `--mode real-data-multiday`. It uses real session start time, end time, and delivered energy from the 4TU/ElaadNL office parking dataset, while price and base-load uncertainty are still simulated inside this project.

## Paper Draft

The current English LaTeX draft is under `paper/main.tex`, with references in `paper/references.bib`. A local Tectonic compile was used during development:

```powershell
tools/tectonic/tectonic.exe paper/main.tex --outdir paper/build --keep-logs --keep-intermediates
```

Regenerate paper figures with:

```powershell
python scripts/make_all_figures.py
```

The latest checked PDF is `paper/build/main.pdf`. It is an SCI-style two-column draft with 45 bibliography entries.
