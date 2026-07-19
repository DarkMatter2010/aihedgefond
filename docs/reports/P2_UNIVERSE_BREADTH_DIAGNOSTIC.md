# P2 Universe-breadth diagnostic (2026-07-19)

Manual confirmation run (not CI). **Single variable changed:** universe
50 → `BROAD_LIQUID_CANDIDATE_UNIVERSE` (503 requested). Everything else bit-identical
to the Phase-2 stack: production `FEATURE_COLUMNS` (36), LightGBM HPs / seed /
train_end / test_start from `limits.yaml`, horizon forced to **h=2** to match the
documented 50-name Rank-IC reference.

Branch: `build/universe-breadth-diagnostic-2026-07-19`.

## Survivorship (honesty)

`BROAD_LIQUID_CANDIDATE_UNIVERSE` is **current** index membership → mild
survivorship bias. This run is a **relative** 50-vs-broad diagnosis, **not** a
tradeable alpha proof.

## Results (seed=42, once, no post-hoc retune)

| Universum | N | median breadth | OOS Rank-IC | IC | ICIR | vs 0.02 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| broad_liquid_candidate | 496 (of 503) | 494.0 | **0.014808** | 0.009397 | 0.059549 | **below** |
| reference_50_h2 (documented) | 50 | n/a | **0.0176** | (documented) | n/a | reference |

Dropped at ingest (quality / missing Yahoo): 7 symbols (incl. FDXF, HONA, AMCR,
HUBB, SMCI, SW, VRT).

## Interpretation rule

- Rank-IC broad >= 0.02 **and** clearly > 50-name → breadth noise, not dead.
- Rank-IC broad still ~0 / not above narrow → features dead.
- Signs: report both Pearson IC and Rank-IC (no forced conclusion from sign alone).

**Verdict: `features_dead`.** Broad Rank-IC (0.0148) is **not** >= 0.02 and is
**not** above the 50-name h=2 reference (0.0176) — slightly lower. Breadth did
not rescue the signal; next step is feature-class expansion from OHLCV
(reversal / low-vol / illiquidity), not more names.

## Research-trial accounting

This diagnostic **counts as a further research trial** for DSR `n_trials`
(universe configuration probe on the same feature stack). When the gate-hardening
lands, bump the conservative headcount accordingly (do not pretend this run was
free).

## Repro

```bash
python scripts/run_universe_breadth_diagnostic.py
```
