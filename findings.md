# Findings

This file records extracted facts from the project materials.

## Project Inventory
- Repository currently contains paper-planning artifacts only: `最终SCI论文结构及建议.md`, `第一版论文结构.md`, `第一版摘要.docx`, and four requirement screenshots under `论文要求/`.
- No experiment code, data generator, notebooks, result tables, or figures are present in the current workspace.
- The project therefore needs an experiment implementation phase before manuscript drafting can begin.

## Paper Requirements
- Topic can be self-selected but must be specific; current EV charging optimization topic fits the "optimization for large-scale problems" and ADMM/distributed optimization direction.
- The project must be completed by one student only.
- The report must be written in English.
- Plagiarism must be avoided.
- Report length must be less than 8 single-column pages with 11 pt font.
- IEEE paper style is suggested; LaTeX with IEEEtran is preferred, specifically `\documentclass[draftcls,onecolumn]{IEEEtran}`.
- Required structure and scoring shown in screenshots:
  - Introduction: 10 pts; describe the problem clearly.
  - Existing Works: 20 pts; read at least 5 papers, overview them with clear logic and unified formulation, comment on contributions and drawbacks.
  - New Contribution: 10 pts; describe technical challenges and improve existing approaches.
  - Numerical Results: 10 pts; compare existing works and evaluate the proposed algorithm.
  - Conclusions.
  - References with a unified style.
- Schedule screenshot says topic plus brief abstract deadline is April 12 and full report deadline is April 30. The current environment date is 2026-04-30, so execution should prioritize experiments and final report assembly.

## Structure and Draft Gaps
- `第一版论文结构.md` frames the work as offline/distributed fair EV charging using convex optimization, centralized benchmark, dual decomposition, and ADMM.
- `第一版摘要.docx` matches that first-version direction: synthetic Python experiments, metrics of total cost, peak load, convergence speed, and fairness; baselines of centralized optimization, dual decomposition, and ADMM.
- `最终SCI论文结构及建议.md` upgrades the paper to an online stochastic setting with random arrivals, virtual queues, Lyapunov drift-plus-penalty, risk-aware capacity buffer, and ADMM as the per-slot distributed solver.
- The main structural gap is that the final SCI version needs online state evolution, deadline-aware virtual queues, V-parameter trade-off experiments, risk-buffer experiments, and deadline/backlog metrics that are absent from the first version.

## Experiment Plan Inputs
- Proposed method: online risk-aware Lyapunov-ADMM.
- Required baselines from the final structure: uncontrolled charging, greedy price-based charging, offline centralized optimum, offline ADMM, dual decomposition, and proposed online Lyapunov-ADMM.
- Required metrics: total charging cost, peak load or peak-to-average ratio, deadline violation rate, average remaining demand/backlog, fairness index, runtime, and ADMM iterations.
- Required scenario axes: EV counts such as 50/100/200/500, 24 or 48 time slots, random arrivals, random parking duration/departure times, random energy demand, time-varying prices, base load noise, and risk buffer sensitivity.

## Experiment Implementation Findings
- The current experiment script is `experiments/ev_charging_experiments.py`.
- Implemented methods: uncontrolled capped charging, greedy deadline/price heuristic, offline centralized LP, dual decomposition, offline ADMM, and online Lyapunov-ADMM.
- A deadline safety floor was added to online Lyapunov-ADMM because the pure soft queue version was too conservative in tight capacity scenarios.
- Key result pattern so far: offline centralized and offline ADMM achieve lower cost using full future information; online Lyapunov-ADMM achieves lower peak load and better capacity safety under online uncertainty.
- Public-data check now uses the 4TU/ElaadNL office parking-lot dataset, DOI `10.4121/80ef3824-3f5d-4e45-8794-3b8791efbd13.v2`.
- The real-data experiment uses real session start time, end time, and delivered energy. It still simulates electricity price and base-load uncertainty, so it is an external-validity check rather than a full field deployment.
- On five high-activity public-data days with up to 50 sessions/day and `V=0.8, kappa=1.0`, online Lyapunov-ADMM reduces mean peak load from 208.26 kW (offline centralized LP) to 198.49 kW and capacity violation from 3.8% to 1.2%, but has 10.8% deadline violation. Lower `V` removes deadline violation but worsens peak/capacity risk.
- The current paper draft compiles with Tectonic to `paper/build/main.pdf` and has 8 pages with 45 cited bibliography entries.
