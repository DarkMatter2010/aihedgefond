# Live CPCV/DSR gate + permutation-null report (2026-07-17)

Manual Yahoo run (not CI). Candidate: horizon=2, 50 symbols, feature-set #17, seed=42.

## Corrected gate (after DSR aggregation fix)

| Field | Value |
| --- | --- |
| aggregation | merge OOS path returns (mean per timestamp) |
| T (n_obs) | 2704 |
| path_sharpe_mean | ≈ 0.0842 |
| observed_sharpe | ≈ 0.0988 |
| var_trial_sharpes | ≈ 0.01274 (research-trial IC→SR proxy; not CPCV path var) |
| sr0 | ≈ 0.178 (older IC table run) / consistent scale non-ann. |
| **corrected DSR** | **≈ 1.09e-6** (was ≈ 0.941 before fix) |
| gate threshold | DSR >= 0.95 |
| **gate verdict** | **NEIN** |

## Permutation null (M=100, seed=20260717)

Cross-sectional label shuffle per date; full gate each time.

| Field | Value |
| --- | --- |
| null DSR q50 | ≈ 3.1e-23 |
| null DSR q95 | ≈ 1.8e-16 |
| real percentile vs null | 100% (beats null numerically) |
| null at ~0.9+? | **NO** — gate no longer rubber-stamps noise |
| beats null 95% + DSR>=0.95? | NO (absolute DSR ≪ 0.95) |
| **corrected verdict / Slice-2 handoff** | **NEIN / BLOCKED** |

## Handoff

Slice 2 (vectorbt / NautilusTrader / Paper) remains **BLOCKED**. Return to Phase-2 signal search; do not invest in the execution stack on this candidate.
