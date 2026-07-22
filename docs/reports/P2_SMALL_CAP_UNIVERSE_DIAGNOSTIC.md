# P2 Small-cap universe diagnostic — ALL_NEW on Russell-2000-segment (2026-07-21)

Manual confirmation run (not CI). **No CPCV / Gate** — Rank-IC triage only.
Universum: SMALL_CAP_CANDIDATE_UNIVERSE (220 requested, **196 kept**, 24 dropped,
drop_rate **0.1091**). Features: ALL_NEW_FEATURE_CLASS_COLUMNS (15). Seed /
	rain_end / configured 	est_start / LightGBM HPs from limits.yaml.
Horizons **h=2** and **h=21** (h=21 	est_start auto-extended to 2023-02-02).
Production FEATURE_COLUMNS (36) **unchanged**.

Script: python scripts/run_small_cap_universe_diagnostic.py (once, seed=42, no
post-hoc retune after seeing results).

## Survivorship (honesty)

SMALL_CAP_CANDIDATE_UNIVERSE is a **static** liquid US small/mid snapshot
outside the S&P-near broad list (Russell-2000-segment style). Current membership
plus small-cap delistings **intensify** survivorship bias vs large-cap. Relative
diagnosis only — **not** a tradeable alpha proof.

## Results (seed=42, once)

| Universum | h | test_start | N_dates | median breadth | Rank-IC | IC | ICIR | vs 0.02 | broad all_new Rank-IC |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| small-cap ALL_NEW | 2 | 2023-01-10 | 745 | 196.0 | **-0.005316** | 0.002121 | 0.012218 | below | 0.0221 |
| small-cap ALL_NEW | 21 | 2023-02-02 | 710 | 196.0 | **-0.006225** | 0.020903 | 0.103817 | below | 0.0348 |

Dropped at ingest (quality / missing Yahoo): 24 / 220 (**10.91%**) — ABR, AHH,
AL, AMED, APLS, ASGN, ASTH, ASTS, AT, ATGE, AXSM, BHLB, BPMC, BRKL, BRY, BXC,
CABO, CADE, CIVI, CLSK, CSGS, CWAN, DGII, ESAB. (Higher than broad ~1.4% as
expected under stale_bars=5 / max_nan_ratio=0.0.)

Best cell Grinold SR (trial log): h=2 Rank-IC -0.005316 × √196 → **-0.074422**.

## Interpretation rule

- Rank-IC >= 0.02 on **any** horizon → candidate for a separate gate prompt.
- Both < 0.02 → **Gratis-OHLCV signal search exhausted** (large + small
  universe, all feature classes, meta-labeling done). No further free-data
  OHLCV variants.

**Verdict: ree_ohlcv_search_exhausted.** Both horizons are **negative**
Rank-IC and far below 0.02. The small-cap lever does not rescue the free OHLCV
stack; the broad ll_new cells (0.0221 / 0.0348) remain the best IC triage
outcomes, and those already failed the hardened gate elsewhere.

## Research-trial accounting

Counts as **one** research trial (two horizons measured; best-horizon Grinold
logged). Appended to RESEARCH_TRIAL_SHARPES: **-0.074422**.

**N_RESEARCH_TRIALS = 25** (= 24 prior + 1 small-cap diagnostic). Derivation
documented in src/aihedgefund/research/research_trials.py.

## Recommendation

End of free OHLCV signal search. Do **not** invent further gratis Yahoo/OHLCV
feature or universe variants. Next work is outside this Free-Data lever
(pipeline Durchstich / paid data / non-OHLCV signals) — not another Phase-2
OHLCV probe.

## HANDOFF FOR NEXT PHASE

| Item | Path / signature |
| --- | --- |
| Universe | SMALL_CAP_CANDIDATE_UNIVERSE, SMALL_CAP_SURVIVORSHIP_BIAS_NOTE |
| Diagnostic API | 
un_small_cap_universe_diagnostic(bars, settings, *, n_symbols_dropped=None) |
| Settings helper | settings_for_small_cap_universe_diagnostic(settings, *, horizon) |
| Manual script | scripts/run_small_cap_universe_diagnostic.py |
| Features used | ALL_NEW_FEATURE_CLASS_COLUMNS (15) — production stack unchanged |
| Trial count | N_RESEARCH_TRIALS = 25 |
| End state | **Free-OHLCV Suchstopp** — no further free-data OHLCV variants |

`ash
python scripts/run_small_cap_universe_diagnostic.py
`
