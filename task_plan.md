# Energy Optimize Paper Task Plan

## Goal
Analyze the project materials and plan the remaining work for an SCI-style paper, using `最终SCI论文结构及建议.md` and the requirement screenshots as the main source of truth. Experiments must be completed before paper drafting.

## Phases

| Phase | Status | Output |
|---|---|---|
| 1. Inventory project materials | complete | File list and source priority |
| 2. Extract paper requirements | complete | Requirement checklist from screenshots and docs |
| 3. Compare existing first-version materials | complete | Gaps between first draft and final SCI structure |
| 4. Define experiment completion plan | complete | Experiments, metrics, figures, tables, and execution order |
| 5. Define paper-writing plan | complete | Chapter-by-chapter writing sequence after experiments |
| 6. Strengthen experiment credibility | complete | Multi-seed statistics, stronger metrics, ablation, risk correlation, ADMM tuning |
| 7. Improve reproducibility packaging | complete | README, .gitignore, pinned requirements, final commands |
| 8. Convert stable results into paper-ready evidence | complete | LaTeX result tables, stronger related work, basic method properties, Chinese figure/table interpretation index |
| 9. Reconcile remaining recommendations | complete | 30-seed baseline, workplace profile, method-characteristics table, remaining-task matrix |
| 10. Close feasible experiment gaps | complete | Large-scale fast scalability and centralized per-slot ablation |
| 11. Close external validation and submission gaps | complete | 4TU/ElaadNL public-data check, 45 references, 8-page PDF, tracked pyc cleanup |
| 12. Reviewer-facing paper polish | complete | Conservative claims, formal propositions, compact algorithm table, and one clean summary figure |
| 13. Build SCI extended version | complete | Two-column SCI-style draft, drift derivation, offline benchmark, metric formulas, full figure set |
| 14. Reviewer-risk consistency pass | complete | Code-paper objective alignment, `gamma` sensitivity, public-data preprocessing disclosure, reproducibility scripts, and tests |

## Experiment-First Execution Plan

### A. Build Reproducible Simulation Core
- [x] Generate stochastic EV sessions: arrival time, departure time, energy demand, max charging power, and charging efficiency.
- [x] Generate system signals: time-varying electricity price, base load, transformer capacity, and optional base-load forecast error variance.
- [x] Track online state: active EV set, remaining demand queue `Q_i(t)`, deadline urgency weight `w_i(t)`, delivered energy, and violations.

### B. Implement Baselines
- [x] Uncontrolled charging: charge at max power subject to availability and capacity handling.
- [x] Greedy price/deadline heuristic: prioritize low-price slots and urgent vehicles.
- [x] Offline centralized optimum: full future information benchmark.
- [x] Offline ADMM: first-version distributed baseline.
- [x] Dual decomposition: course-aligned decomposition baseline.
- [x] Proposed online Lyapunov-ADMM: final SCI main method.

### C. Produce Required Metrics
- [x] Total charging cost.
- [x] Peak load and peak-to-average ratio.
- [x] Deadline violation rate.
- [x] Average backlog or remaining demand.
- [x] Fairness index.
- [x] Runtime and ADMM iteration count.
- [x] Capacity violation rate for risk-buffer experiments.

### D. Required Figure/Table Set Before Writing
- [x] Scenario/data setup table.
- [x] Baseline comparison table across methods.
- [x] Cost and peak-load comparison plot.
- [x] Deadline violation/backlog comparison plot.
- [x] Cost-delay trade-off plot versus Lyapunov parameter `V`.
- [x] Scalability plot versus number of EVs.
- [x] Risk-buffer sensitivity plot versus `kappa` or `epsilon`.

### E. Stop Condition Before Drafting
- No manuscript drafting begins until the experiment script/notebook can reproduce all final tables and figures from a fixed random seed.
- Current technical baselines are implemented; remaining pre-writing work is result sanity-checking and figure/table selection.

## Source Priority
1. `最终SCI论文结构及建议.md`
2. `论文要求/*.png`
3. `第一版论文结构.md`
4. `第一版摘要.docx`

## Decisions
- The immediate next work is experimental planning and completion, not direct manuscript drafting.
- The final paper direction should follow the final SCI document, not the first-version offline ADMM structure.
- The first-version materials should be reused as baselines and early abstract context, not as the final paper's main contribution.
- The first English draft is written in LaTeX under `paper/`, using IEEEtran one-column style and linking to the generated experiment figures.
- Based on `后续任务及建议.md`, the next priority is not more prose but stronger experiments: multi-seed statistics, ablation, risk correlation, and baseline diagnostics.

- After the enhanced experiments stabilized, the paper draft should use the 10-seed baseline table, refined `V` sweep, risk-correlation sweep, and ablation table as the main evidence set.

- The baseline evidence has been upgraded again from 10 seeds to 30 seeds. Workplace-profile results are used only as sensitivity evidence, not as a claim of real-data validation.
- Public-data validation uses the 4TU/ElaadNL office parking dataset as an external-validity check. It uses real session timing and energy, but simulated price and base-load uncertainty, so it should not be described as a full field deployment.
- To satisfy the 8-page single-column constraint with 45 references, the final course-report draft keeps core tables plus one compact summary figure in the paper, while leaving detailed sweep figures in `outputs/` for presentation and extended versions.
- The public-data result should be described as an external-validity check using real session timing and energy, not as a full real-world charging-station validation.
- The main method should be described as a deployment-oriented online scheduler. It is not designed to dominate offline full-information benchmarks in cost; its value is online operation, lower peak load, capacity safety, and distributed coordination.
- The current submission target has shifted from the 8-page course draft to the SCI extended version. `paper/main.tex` is now the two-column extended manuscript, and the paper is no longer constrained to 8 single-column pages.
- All formal paper figures should be generated by `experiments/make_paper_figures.py`; manual screenshots should not be used for the manuscript.
- The online per-slot objective in the paper must explicitly include the implementation's proximal term `alpha/2 * sum_i p_i(t)^2`.
- The reference load should be described as `P_ref(t)=gamma * [C_t - Bhat_t - kappa sigma_t]^+`; `gamma=0.82` is a moderate operating point supported by a sensitivity sweep, not a universal optimum.
- `uncontrolled_capped` should be called `Immediate capped` in paper-facing text and figures to avoid implying a capacity-violating uncontrolled baseline.
- Public-data experiments should report filtering and clipping statistics; in the current five-day check, 250 sessions are selected and zero session energies are clipped.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| `python-docx` failed on literal Chinese path rendered as `?????.docx` | Directly opened `第一版摘要.docx` in Python source | Re-ran extraction by globbing `*.docx`; content extracted successfully |
| PowerShell profile warning appears before commands | Running PowerShell commands in this environment | Commands still completed when exit code was 0; warning is unrelated to experiment code |
| `pdflatex`/`bibtex` not found | Checked LaTeX compiler availability | Generated LaTeX source only; PDF compilation must be done in a LaTeX-enabled environment |
| Tectonic compile timed out | First run with a new local TeX engine | Re-ran with `--print` after cache warm-up; PDF compiled successfully |
