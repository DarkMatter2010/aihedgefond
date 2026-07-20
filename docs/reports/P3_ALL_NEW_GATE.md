# P3 all_new CPCV/DSR gate + permutation null (2026-07-20)

Manual confirmation run (not CI). **Single seed-fixed Yahoo execution** — no
post-hoc retune / seed fishing.

Candidate from [P2_FEATURE_CLASS_TRIAGE](P2_FEATURE_CLASS_TRIAGE.md):
`ALL_NEW_FEATURE_CLASS_COLUMNS` (15 cols: Reversal + Low-Vol + Parkinson Range),
`BROAD_LIQUID_CANDIDATE_UNIVERSE` (496 kept / 503), seed=42.

Primary horizon **h=21** (triage Rank-IC 0.0348). Secondary **h=2** reported for
context only. CPCV knobs unchanged from prior hardened gate (N=6, k=2).

This validation run does **not** increment `N_RESEARCH_TRIALS` (already counted
in the feature-class triage table; `n_trials=23`).

Script: `python scripts/run_gate_on_all_new_features.py`

## Survivorship (honesty)

`BROAD_LIQUID_CANDIDATE_UNIVERSE` is current index membership → mild survivorship
bias. Not a tradeable alpha proof.

## Primary gate — h=21 (`run_overfitting_gate`)

| Field | Value |
| --- | --- |
| features | 15 (`ALL_NEW_FEATURE_CLASS_COLUMNS`) |
| n_symbols | 496 |
| aggregation | `merge_cpcv_path_returns` |
| T (`n_obs`) | 2682 |
| path_sharpe_mean | ≈ 0.2146 |
| observed_sharpe | ≈ 0.2224 |
| `var_trial_sharpes` | ≈ 0.1020 (`RESEARCH_TRIAL_SHARPES`, not CPCV path var) |
| `n_trials` | **23** |
| sr0 | ≈ 0.6264 |
| **DSR** | **≈ 6.92e-112** |
| gate threshold | DSR >= 0.95 |
| **gate verdict** | **NEIN** |

## Permutation null — h=21 (M=100, seed=20260717)

Cross-sectional label shuffle per date; full gate each time.

| Field | Value |
| --- | --- |
| null DSR q50 | ≈ 1.39e-230 |
| null DSR q95 | ≈ 8.01e-208 |
| real percentile vs null | 100% (beats null numerically) |
| null at ~0.9+? | **NO** |
| DSR >= 0.95? | **NO** |
| **corrected verdict / Slice-2 handoff** | **NEIN / BLOCKED** |

## Secondary gate — h=2 (context only)

| Field | Value |
| --- | --- |
| observed_sharpe | ≈ 0.0708 |
| T | 2701 |
| **DSR** | **≈ 1.55e-196** |
| verdict | **NEIN** |

## Interpretation

**Signal not validated.** Path Sharpes are weakly positive, but Bailey DSR under
`n_trials=23` and the logged research-trial variance collapses to ~0 — far below
0.95. Beating the permutation null is necessary but not sufficient; absolute DSR
fails.

### Recommendation

**End free-yfinance OHLCV feature hunting** on this large-cap / survivorship-biased
dataset. Documented project standing: **no CPCV/DSR-validated tradeable signal**.
Do **not** start Phase 4 without a validated signal. Do not fish with other seeds,
horizons, or CPCV knobs on this candidate.

## HANDOFF FOR NEXT PHASE

| Item | Value |
| --- | --- |
| Verdict | **NEIN** — no Slice-2 execution handoff |
| Feature set tested | `ALL_NEW_FEATURE_CLASS_COLUMNS` via `aihedgefund.research.all_new_gate` |
| Script | `scripts/run_gate_on_all_new_features.py` |
| Next (if continuing research) | New data sources / universes / costly features outside gratis OHLCV — **not** another OHLCV class sweep on this stack |
| Next (if shipping) | Pipeline Durchstich with documented weak signal only as engineering exercise, not as validated alpha |
