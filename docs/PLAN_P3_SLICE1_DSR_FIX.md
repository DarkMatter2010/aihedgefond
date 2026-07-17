# Plan — Phase 3 Slice 1 FIX: CPCV/DSR-Gate Härtung

**Branch:** `build/cpcv-dsr-gate-2026-07-17`  
**Scope:** Produktionscode + Tests + Live-Null-Report. Kein Slice 2 (vectorbt/Nautilus/Paper).

## Root cause (bestätigt am Code)

In `research/gate.py` wird `var_trial_sharpes` aus der **Varianz der CPCV-Pfad-Sharpes** geschätzt. Die C(N,k) Pfade sind **nicht unabhängig** (überlappende Trainingsfenster). Kleine Pfadvarianz → SR0 ≈ 0 → DSR ≈ Φ(SR̂·√(T−1)) ≈ 0.94 auch bei Rank-IC≈0.018 (Rauschen). Zusätzlich: T ist bereits die Länge der aggregierten Renditereihe (korrekt), aber die Quelle von `var_trial_sharpes` ist falsch.

## Ansatz (explizit)

**OOS-Renditen zu EINER Reihe zusammenführen** (Mean über überlappende Timestamps), dann **EINEN** nicht-annualisierten Sharpe mit `T = n_obs = Anzahl Renditebeobachtungen` bilden.  
`var_trial_sharpes` / SR0 kommen **nur** aus der Varianz der ~12 echten Research-Trial-Sharpes (IC→SR-Proxy dokumentiert) — nie aus CPCV-Pfadvarianz. SR̂ und SR0 bleiben nicht-annualisiert (gleiche Skala).

## Dateien / Reihenfolge

1. `src/aihedgefund/research/deflated_sharpe.py` — T / `var_trial_sharpes`-Doku härten
2. `src/aihedgefund/research/trial_meta.py` — **neu**: dokumentierte Trial-ICs → nicht-ann. Sharpes → Varianz
3. `src/aihedgefund/research/gate.py` — `aggregate_cpcv_path_returns`; `var_trial_sharpes` als Pflicht-Parameter; Pfadvarianz nur Diagnostik
4. `scripts/run_cpcv_dsr_gate.py` — Trial-Varianz verdrahten; Report-Felder
5. `scripts/run_gate_permutation_null.py` — **neu**: M≥100 Label-Permutationen, Null-Quantile, Perzentil, Akzeptanz
6. `tests/test_phase3.py` — Signaturen + Known-Noise über echten Aggregations-Pfad
7. Live: Yahoo 1× Gate + Permutations-Null → Report
8. ruff / mypy / pytest → PR (kein Auto-Merge)

## Nicht im Scope

vectorbt, NautilusTrader, Paper-Trading, mlfinlab, Merge nach main.
