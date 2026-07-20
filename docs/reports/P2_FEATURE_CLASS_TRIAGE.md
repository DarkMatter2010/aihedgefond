# P2 Feature-class IC triage — Reversal / Low-Vol / Parkinson Range (2026-07-19)

Manual confirmation run (not CI). **No CPCV / Gate** — Rank-IC triage only.
Universum: `BROAD_LIQUID_CANDIDATE_UNIVERSE` (503 requested, 496 kept). Seed /
`train_end` / configured `test_start` / LightGBM HPs from `limits.yaml`.
Horizons **h=2** and **h=21** (h=21 `test_start` auto-extended to 2023-02-02 so
trading-bar label gap > horizon). Production `FEATURE_COLUMNS` (36) **unchanged**;
new classes measured as separate column blocks.

Script: `python scripts/run_feature_class_triage.py` (once, seed=42, no post-hoc
retune after seeing results).

## Survivorship (honesty)

`BROAD_LIQUID_CANDIDATE_UNIVERSE` is **current** index membership → mild
survivorship bias. Relative diagnosis only — **not** a tradeable alpha proof.

## Reversal sign convention

`reversal_k = -1 * past_k_day_return`. A **positive** OOS Rank-IC means the
Reversal hypothesis carries (recent losers rank higher for the forward label).

## Results (seed=42, once)

| Klasse | h | test_start | n_feat | N_dates | median breadth | Rank-IC | IC | ICIR | vs 0.02 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| reversal | 2 | 2023-01-10 | 6 | 745 | 494.0 | 0.010769 | 0.014181 | 0.117920 | below |
| reversal | 21 | 2023-02-02 | 6 | 710 | 494.0 | 0.008965 | 0.028614 | 0.233491 | below |
| low_vol | 2 | 2023-01-10 | 6 | 745 | 494.0 | 0.016887 | 0.023327 | 0.143864 | below |
| low_vol | 21 | 2023-02-02 | 6 | 710 | 494.0 | **0.048502** | 0.072018 | 0.380011 | **>=** |
| range_vol | 2 | 2023-01-10 | 3 | 745 | 494.0 | 0.003369 | 0.010942 | 0.089542 | below |
| range_vol | 21 | 2023-02-02 | 3 | 710 | 494.0 | 0.033576 | 0.067277 | 0.357580 | **>=** |
| all_new | 2 | 2023-01-10 | 15 | 745 | 494.0 | **0.022127** | 0.021525 | 0.128726 | **>=** |
| all_new | 21 | 2023-02-02 | 15 | 710 | 494.0 | **0.034849** | 0.058842 | 0.342450 | **>=** |
| new_plus_old | 2 | 2023-01-10 | 51 | 745 | 494.0 | 0.015068 | 0.019052 | 0.122814 | below |
| new_plus_old | 21 | 2023-02-02 | 51 | 710 | 494.0 | 0.022854 | 0.047260 | 0.345767 | **>=** |

Dropped at ingest (quality / missing Yahoo): 7 symbols (FDXF, HONA, AMCR, HUBB,
SMCI, SW, VRT).

## Interpretation rule

- Any class/combo with OOS Rank-IC >= 0.02 **and** stable on both h=2 and h=21
  → candidate for the hardened gate (`DSR >= 0.95`).
- All < 0.02 → OHLCV/large-cap dead; pipeline Durchstich instead of more features.

**Verdict: `candidate_for_gate`.** Class **`all_new`** (Reversal + Low-Vol +
Range-Vol together) clears Rank-IC >= 0.02 on **both** horizons
(h=2: 0.0221, h=21: 0.0348). Best single cell: `low_vol` h=21 Rank-IC
**0.0485** (not stable alone on h=2: 0.0169).

### Reversal sign finding

Reversal Rank-IC is **positive** on both horizons (0.0108 / 0.0090) → direction
matches the Reversal hypothesis, but magnitude stays **below** 0.02. Reversal
alone is not the candidate; it contributes inside `all_new`.

## Research-trial accounting

Every measured config counts. Appended to `RESEARCH_TRIAL_SHARPES`:

1. Prior unlogged universe-breadth diagnostic (Rank-IC 0.014808 × √494 → 0.329)
2. Ten triage Grinold SRs from this run (Rank-IC × √494)

**`N_RESEARCH_TRIALS = 23`** (= 12 legacy + 1 breadth + 10 triage). Derivation
documented in `src/aihedgefund/research/research_trials.py`.

## Recommendation

Next phase: run **`all_new`** (15 columns =
`ALL_NEW_FEATURE_CLASS_COLUMNS`) through the hardened overfitting gate
(`run_overfitting_gate`, DSR >= 0.95) on the broad universe. Do **not** hunt
more free OHLCV features until that gate result is known.

## HANDOFF FOR NEXT PHASE

| Item | Path / signature |
| --- | --- |
| Feature registries | `aihedgefund.features.feature_classes` |
| Column sets | `REVERSAL_FEATURE_COLUMNS`, `LOW_VOL_FEATURE_COLUMNS`, `RANGE_VOL_FEATURE_COLUMNS`, `ALL_NEW_FEATURE_CLASS_COLUMNS`, `NEW_PLUS_OLD_FEATURE_COLUMNS`, `FEATURE_CLASS_CONFIGS` |
| Matrix builder | `build_triage_feature_matrix(bars, parameters=None) -> DataFrame` |
| Indicators | `reversal`, `inverse_realized_vol`, `parkinson_volatility` in `features/indicators.py` |
| Triage API | `run_feature_class_triage(bars, settings) -> FeatureClassTriageReport` |
| Settings helper | `settings_for_feature_class_triage(settings, *, horizon) -> Settings` |
| Manual script | `scripts/run_feature_class_triage.py` |
| Gate candidate | label=`all_new`, columns=`ALL_NEW_FEATURE_CLASS_COLUMNS`, horizons already cleared at IC triage |
| Production stack | `FEATURE_COLUMNS` still 36 — **do not** silently swap until gate passes |

```bash
python scripts/run_feature_class_triage.py
```
