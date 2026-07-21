# P2 Meta-Labeling Triage (Triple-Barrier + SMA primary)

**Date:** 2026-07-20  
**Script:** `scripts/run_meta_labeling_triage.py`  
**Seed:** 42 (from `limits.yaml`)  
**Universe:** `BROAD_LIQUID_CANDIDATE_UNIVERSE` (requested 503, kept **496**, dropped 7)  
**Gate:** not run in this step

## Setup (pre-registered, no retune)

| Knob | Value |
|------|--------|
| Primary | SMA-10 sign (`+1` / `-1`) |
| Barriers | `pt=1.0`, `sl=1.0`, `vertical_bars=10`, `vol_span=20` |
| Meta features | `ALL_NEW_FEATURE_CLASS_COLUMNS` (15) |
| Meta model | LightGBM `objective=binary`, HPs from `limits.yaml`, accept if `P(win) >= 0.5` |
| Split | `train_end=2022-12-30`, effective `test_start=2023-01-20` (embargo ≥ 10) |

Survivorship: current-index membership (same note as prior broad-universe runs).

## OOS results

| Metric | Value |
|--------|--------|
| n_train / n_test | 922 513 / 359 632 |
| n_accepted (accept_rate) | 185 823 (0.5167) |
| Precision | **0.530526** |
| Recall | 0.549410 |
| F1 | 0.539803 |
| Base rate (primary wins) | 0.498943 |
| Lift (precision / base) | 1.063 |
| Confusion | TP=98584 FP=87239 FN=80852 TN=92957 |
| Primary bet Sharpe OOS | −0.004417 |
| Filtered bet Sharpe OOS | **0.049881** |

## Interpretation

- Precision clears base rate by > 0.02 absolute margin.
- Filtered OOS bet Sharpe beats unfiltered primary.

**Verdict: `candidate_for_gate`**

Next step (separate prompt): run hardened CPCV/DSR gate (`DSR ≥ 0.95`) on this meta-labeling configuration. Do not retune MA / barriers / threshold first.

## Research-trial accounting

- Counts as **1** new research trial (primary rule + meta config).
- Logged Sharpe (row 24): **0.049881** = filtered OOS per-bet Sharpe.
- `N_RESEARCH_TRIALS`: 23 → **24**.

## HANDOFF

- Module: `aihedgefund.research.meta_labeling`
- Report: this file
- Next phase: CPCV/DSR gate on meta-labeling candidate (not feature re-hunt)
