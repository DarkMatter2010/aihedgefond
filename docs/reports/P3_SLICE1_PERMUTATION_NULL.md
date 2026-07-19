# P3 Slice 1 — CPCV/DSR gate + permutation null (2026-07-17)

Manual confirmation run (not CI). Candidate: horizon=2, 50 symbols, feature-set #17, seed=42.
Branch: `build/gate-hardening-onto-main-2026-07-17` (landed from `510123e` + `n_trials=15`).

## Gate (`scripts/run_cpcv_dsr_gate.py`)

| Field | Value |
| --- | --- |
| aggregation | `merge_cpcv_path_returns` (mean per timestamp) |
| T (`n_obs`) | 2704 |
| path_sharpe_mean | ≈ 0.0842 |
| observed_sharpe | ≈ 0.0988 |
| `var_trial_sharpes` | ≈ 0.01227 (`RESEARCH_TRIAL_SHARPES`, not CPCV path var) |
| `n_trials` | **15** (conservative; 12 logged ICs rounded up) |
| sr0 | ≈ 0.196 |
| **DSR** | **≈ 1.14e-7** |
| gate threshold | DSR >= 0.95 |
| **gate verdict** | **NEIN** |

## Permutation null (`scripts/run_gate_permutation_null.py`, M=100, seed=20260717)

Cross-sectional label shuffle per date; full gate each time.

| Field | Value |
| --- | --- |
| null DSR q50 | ≈ 4.0e-25 |
| null DSR q95 | ≈ 4.7e-18 |
| real percentile vs null | 100% (beats null numerically) |
| null at ~0.9+? | **NO** |
| DSR >= 0.95? | **NO** |
| **corrected verdict / Slice-2 handoff** | **NEIN / BLOCKED** |

## Notes

- Run once; no post-hoc retuning after seeing results.
- `n_trials=15` derivation: documented configs (Baseline, CA-Fix, Feature-Set #17, Sweep h=1/2/5/10/20, Mom-Breadth h=63/126 + diagnostics) ≥10 logged; 12 IC rows in the variance table; round defensively up to 15 for unlogged probes.
