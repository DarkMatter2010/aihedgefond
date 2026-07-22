# P3 Gate — insider_plus_all_new CPCV/DSR (2026-07-22)

Manual confirmation run (not CI). **One shot, seed=42**, no retune / seed fishing.
Universum: **BROAD_LIQUID_CANDIDATE_UNIVERSE** (same as triage Rank-IC 0.0403;
503 requested, **496 kept**, 7 dropped). Features: INSIDER_PLUS_ALL_NEW (24).
Primary **h=21**, secondary h=2 (context). CPCV N=6 k=2. M=100 label permutation
null. `N_RESEARCH_TRIALS=29` (does **not** increment — triage already counted).

Script: `python scripts/run_gate_on_insider_combo.py`

## Results

| Item | Value |
| --- | --- |
| Universe | BROAD_LIQUID (kept 496) |
| Form4 rows | 218777 (305 symbols without filings → neutral 0) |
| Primary h | 21 |
| path_sharpe_mean / std | 0.2133 / 0.0800 |
| observed_sharpe | 0.2259 |
| n_trials | 29 |
| var_trial_sharpes | 0.1041 |
| **DSR (primary)** | **2.47e-139** |
| Gate threshold | 0.95 |
| Gate verdict (raw) | **NEIN** |
| null DSR q50 / q95 | 2.01e-256 / 3.12e-230 |
| real percentile vs null | 100.00 (beats null q95) |
| beats_null_q95 | True |
| gate_dsr_ge_0_95 | False |
| **corrected_verdict** | **NEIN** |
| Secondary h=2 DSR | 4.62e-230 (NEIN; context only) |

Triage Rank-IC was 0.0204 / 0.0403 — the combo cleared IC triage but **fails**
the hardened DSR gate under selection bias with n_trials=29. Consistent with
interaction-overfitting suspicion (combo > parts).

## Interpretation

**SEARCH_STOP — free-data signal search permanently ended.**

Exhausted levers:
1. OHLCV large-cap (feature classes, all_new gate NEIN)
2. OHLCV small-cap universe diagnostic (negative IC)
3. Meta-labeling (gate NEIN)
4. SEC Form 4 insider alone (weak IC) + insider_plus_all_new (gate NEIN)

No further free-data Phase-2/3 signal-hunt prompts.

## Project close-out decision (not another signal hunt)

| Option | Meaning |
| --- | --- |
| A — Pipeline deliverable | Document the research pipeline as the product (ingest → features → triage → CPCV/DSR gate) and stop alpha search |
| B — Paid data | Budget discussion for non-free vendors (fundamentals, alternatives) before any new signal work |

## HANDOFF FOR PROJECT DECISION

| Item | Path / value |
| --- | --- |
| Module | `aihedgefund.research.insider_combo_gate` |
| Script | `scripts/run_gate_on_insider_combo.py` |
| Feature set | INSIDER_PLUS_ALL_NEW_FEATURE_COLUMNS (24) |
| Universe | BROAD_LIQUID_CANDIDATE_UNIVERSE |
| Primary DSR | 2.47e-139 |
| Corrected verdict | NEIN |
| Trial count | N_RESEARCH_TRIALS = 29 (unchanged by this gate) |
| Next | Choose A or B above — **not** another free-data feature/universe/meta prompt |

`ash
python scripts/run_gate_on_insider_combo.py
`
