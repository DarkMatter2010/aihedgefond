# P2 Insider Form 4 IC triage — SEC EDGAR (2026-07-22)

Manual confirmation run (not CI). **No CPCV / Gate** — Rank-IC triage only.
Universum: BROAD_LIQUID_CANDIDATE_UNIVERSE (503 requested, **496 kept**, 7 dropped).
Features: insider alone (9) and insider_plus_all_new (24). Seed / train_end /
configured test_start / LightGBM HPs from limits.yaml. Horizons **h=2** and **h=21**.
Production FEATURE_COLUMNS (36) **unchanged**.

Script: python scripts/run_insider_form4_triage.py (once, seed=42, no post-hoc retune).

## PIT note

Form 4 features use **filed_at** (SEC cceptanceDateTime) only.
	ransaction_date is informational and may precede filing by up to ~2 business days.

## Live data honesty (SEC)

- Official endpoints only (data.sec.gov submissions + www.sec.gov Archives XML /
  company_tickers). No third-party wrappers.
- Rate-limited (edgar.max_rps=8), disk-cached under .cache/sec_edgar/.
- Live script used include_historical_files=False (submissions 
ecent window) and
  skip_uncached_filings=True after a long warm cache build so the registered run
  finishes in wall-clock; **218777** transaction rows retained after dedupe;
  **305** symbols without parsed Form 4 rows in-cache (neutral 0 features, not a
  hard-fail). Adapter supports full historical iles[] + live XML fetch when
  include_historical_files=True and skip_uncached_filings=False.

## Results (seed=42, once)

| Config | h | test_start | n_feat | N_dates | median breadth | Rank-IC | IC | ICIR | GrinoldSR | vs 0.02 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| insider | 2 | 2023-01-10 | 9 | 745 | 494.0 | **0.001150** | 0.001410 | 0.029041 | 0.025551 | below |
| insider | 21 | 2023-02-02 | 9 | 710 | 494.0 | **0.005343** | 0.012315 | 0.243740 | 0.118756 | below |
| insider_plus_all_new | 2 | 2023-01-10 | 24 | 745 | 494.0 | **0.020408** | 0.019559 | 0.127553 | 0.453581 | above |
| insider_plus_all_new | 21 | 2023-02-02 | 24 | 710 | 494.0 | **0.040304** | 0.064024 | 0.374875 | 0.895811 | above |

Prior best free-OHLCV triage (broad ll_new): Rank-IC **0.0221 / 0.0348** (h=2 / h=21).
Best insider cell: **insider_plus_all_new h=21 Rank-IC 0.040304** (above prior all_new).

## Interpretation rule

- Rank-IC >= 0.02 on any measured cell → candidate for a separate gate prompt.
- All cells < 0.02 → free-data levers exhausted.

**Verdict: candidate_for_gate.** Insider alone is weak (<0.02). Combined with
ALL_NEW it clears the threshold on both horizons and beats the prior broad
ll_new IC on h=21. Next step is the hardened CPCV/DSR gate — not another
free-data feature hunt.

## Research-trial accounting

Counts as **four** research trials (2 configs × 2 horizons), cell-level like
feature-class triage. Appended Grinold Sharpes:

| # | Cell | Grinold |
| ---: | --- | ---: |
| 26 | insider h=2 | 0.025551 |
| 27 | insider h=21 | 0.118756 |
| 28 | insider_plus_all_new h=2 | 0.453581 |
| 29 | insider_plus_all_new h=21 | 0.895811 |

**N_RESEARCH_TRIALS = 29** (= 25 prior + 4 insider cells). Derivation in
src/aihedgefund/research/research_trials.py.

## Recommendation

Do **not** invent further gratis feature variants before gating.
Run **insider_plus_all_new** through the Phase-3 overfitting gate (DSR >= 0.95).
If the gate is NEIN, free-data levers (OHLCV large+small, feature classes, meta,
Form 4) are exhausted for production alpha claims.

## HANDOFF FOR NEXT PHASE

| Item | Path / signature |
| --- | --- |
| Adapter | SecEdgarForm4Provider in data/adapters/sec_edgar.py |
| Quality | Form4QualityGate (missing activity = neutral; source-down = fail) |
| Features | eatures/insider.py — filed_at PIT; fillna(0) after CS |
| Triage API | 
un_insider_form4_triage(bars, form4, settings) |
| Manual script | scripts/run_insider_form4_triage.py |
| Trial count | N_RESEARCH_TRIALS = 29 |
| Next | Gate prompt on insider_plus_all_new (no auto-merge) |

`ash
python scripts/run_insider_form4_triage.py
`
