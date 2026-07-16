# Implementation Roadmap: AI-Hedge-Fund (Solo-Entwickler, Live-Eigenkapital, Aktien+FX)

> Basis: `Referenzarchitektur_eines_modernen_vollautomatisierten_institut.md`
> Kontext dieser Roadmap: **Solo-Entwickler**, Ziel **Live-Trading mit eigenem Kapital** (nicht institutionell/Fremdkapital), Asset-Fokus **Aktien + FX**.
>
> **Wichtige Abweichung von der Recherche:** Die Original-Recherche ist für ein Team/institutionelles Setup geschrieben. Diese Roadmap streicht bzw. verschiebt Komponenten, die für Investoren-Rechenschaft gedacht sind (volle Compliance-Reporting-Engine, Capacity-Management, Multi-Tenant-Restricted-Lists) und ersetzt schwere Infrastruktur (Feast, Celery/Redis, Prefect/Dagster) im MVP durch einfachere Lösungen. Diese Entscheidungen sind unten explizit begründet, nicht stillschweigend getroffen.

---

## Grundprinzipien (nicht verhandelbar, aus der Recherche übernommen)

1. **Backtest-zu-Live-Parität ist Pflicht.** Kein separater Re-Implementierungsschritt zwischen Research und Live-Code.
2. **Overfitting-Kontrolle (CPCV + Deflated Sharpe Ratio) vor jedem Go-Live**, auch bei kleinem Kapital. Eine hohe In-Sample-Sharpe ohne DSR > 0 ist kein Signal, sondern Rauschen.
3. **Risk-Engine mit Pre-Trade-Checks ist Pflicht ab Tag 1 von Live-Trading**, unabhängig von Teamgröße. Eigenkapital ist genauso real wie Fremdkapital.
4. **Jede neue Modellklasse muss die einfache GBDT-Baseline (LightGBM) risikoadjustiert schlagen**, bevor sie in Produktion geht. Kein Transformer/RL ohne bewiesenen Mehrwert.
5. **Lose Kopplung via Message-Bus + Adapter-Pattern**, damit du als Einzelperson später Komponenten austauschen kannst, ohne den Kern anzufassen.

---

## Phasenübersicht

| Phase | Name | Typ | Kernfrage |
|---|---|---|---|
| 0 | Architektur- & Projekt-Fundament | MVP | Wie ist das System strukturiert? |
| 1 | Daten-Fundament | MVP | Sind die Daten sauber und leak-frei? |
| 2 | Research-Kern & Signal-Baseline | MVP | Gibt es überhaupt Alpha? |
| 3 | Backtesting & Overfitting-Kontrolle | MVP | Ist das Signal echt oder Data-Mining? |
| 4 | Portfolio-Optimierung & Risk-Engine | MVP | Wie viel wird riskiert, und wer stoppt es? |
| 5 | Order-Management, Execution & Paper-Trading | MVP | Wie wird gehandelt, ohne Geld zu verbrennen? |
| 6 | Audit, Steuer/Reconciliation & Go-Live-Vorbereitung | MVP | Ist jede Entscheidung nachvollziehbar und steuerlich sauber? |
| 7 | Monitoring, Analytics & Dashboard | MVP | Merkst du, wenn etwas schiefläuft? |
| **— MVP-Gate —** | *Deflated Sharpe > 0 unter CPCV + Paper-Trading-Parität nachgewiesen* | | |
| 8 | FX-Erweiterung | Advanced | Zweite Assetklasse, sauber nachgezogen |
| 9 | Explainability & Multi-Agent-Layer | Advanced/Optional | Verstehst du, warum das System handelt? |
| 10 | Strategy-Registry, Ensemble & Workflow-Orchestrierung | Advanced | Skaliert die Entwicklung über mehrere Strategien? |
| 11 | Marktregime & Sentiment | Advanced | Passt sich das System an Marktphasen an? |
| 12 | Selbstoptimierung & Alpha-Decay-Monitoring | Advanced | Bleibt die Strategie über Zeit profitabel? |
| 13 | RL & Transformer-Forschungsspur | Optional/Research | Lohnt sich der Aufwand exotischer Modelle? |
| Anhang | Institutionelle Erweiterung | Nur bei Fremdkapital | Explizit NICHT Teil dieser Roadmap |

---

## Phase 0 — Architektur- & Projekt-Fundament

**Ziel:** Ein Skelett, in das jede spätere Komponente eingehängt werden kann, ohne den Kern umzubauen.

**Warum zuerst:** Jede andere Phase erzeugt Module, die kommunizieren müssen. Ohne definierten Message-Bus, DTO-Schemas und Verzeichnisstruktur baust du sonst 20 Komponenten, die man später mühsam verdrahten muss.

### Komponenten
- **Projektstruktur (hexagonale/ports-and-adapters-Architektur)**
  - Verzeichnisse: `data/`, `research/`, `signals/`, `portfolio/`, `risk/`, `execution/`, `backtest/`, `audit/`, `dashboard/`, `config/`
  - Klare Trennung: Domänenlogik (Kern) vs. Adapter (Vendors, Broker, DB)
- **Interner Message-Bus (Pub/Sub)**
  - Solo-Version: In-Process-Event-Bus (z. B. `blinker` oder simples Python-Observer-Pattern) statt Redis-Bus — Redis erst, wenn Multi-Prozess/Multi-Host nötig wird
  - Klare Trennung Command (Order) vs. Event (Fill), wie in NautilusTrader-Konzept
- **DTOs/Schemas via Pydantic** für alle Modul-Grenzen (Signal, Order, Fill, RiskCheck, Feature-Vektor)
- **Konfigurations-/Business-Rule-Engine (Skeleton)**
  - YAML/JSON-Konfiguration statt Hardcoding (Trading-Limits, Universe, Feature-Flags)
  - Feature-Flags: einfache Config-Datei statt Redis-backed Feature-Flags (Solo-Kontext)
- **Dependency-Pinning von Tag 1 an** (`requirements.txt`/`poetry.lock`) — Begründung: LangGraph/CrewAI/AutoGen etc. ändern sich laut Recherche schnell mit Breaking Changes
- **Adapter-Pattern-Grundgerüst** für zukünftige Data-Vendor- und Broker-Adapter (abstrakte Basisklassen)
- **Versionskontrolle & CI-Grundgerüst** (Linting, Tests, Type-Checking)

### Definition of Done
- [ ] Repo-Struktur steht, dokumentiert in README
- [ ] Event-Bus lauffähig mit Dummy-Event (z. B. `SignalGenerated` → `LogSubscriber`)
- [ ] Pydantic-Schemas für Signal, Order, Fill, RiskCheck definiert und getestet
- [ ] Config-Datei lädt Trading-Limits und wird von einem Dummy-Modul gelesen
- [ ] Alle Kern-Dependencies mit exakter Version gepinnt
- [ ] CI-Pipeline läuft grün bei jedem Commit

---

## Phase 1 — Daten-Fundament (Data Vendor Integration, Quality Gate, Feature-Pipeline)

**Ziel:** Saubere, point-in-time-korrekte Daten. Ohne das ist jedes spätere Signal wertlos.

**Warum vor Research:** Look-ahead-Bias durch fehlerhafte Zeitstempel ist laut Recherche die häufigste Ursache für nicht-reproduzierbare Backtests. Das muss vor jeder Modellierung stehen.

### Komponenten
- **Data Vendor Integration Layer**
  - Adapter für mind. einen Aktien-Datenanbieter (z. B. yfinance für MVP, später kostenpflichtiger Vendor)
  - Symbol-Mapping, Zeitzonen-Normalisierung
  - Fallback-Mechanismus bei Vendor-Ausfall (auch als Solo wichtig — ein einzelner Ausfall darf keine stille Datenlücke erzeugen)
- **Data Quality Gate**
  - Staleness-Checks, NaN-Ratio-Checks, Outlier-Checks vor jeder Verarbeitung
  - Harte Fehlschläge (kein "stiller" Weiterlauf mit schlechten Daten)
- **Corporate Actions Handling**
  - Splits, Dividenden — Preisbereinigung (kritisch für Aktien-Backtest-Korrektheit)
  - Nutzung vorhandener Handling-Logik aus Zipline-Reloaded-Ökosystem als Referenz
- **Feature-Pipeline mit Point-in-Time-Korrektheit**
  - **Solo-Entscheidung:** Kein volles Feast-Setup im MVP. Stattdessen: eigene, disziplinierte Point-in-Time-Join-Logik in Polars/pandas mit strikten `asof`-Joins und Zeitstempel-Validierungstests. Feast wird in Phase 10 nachgerüstet, falls die Komplexität es rechtfertigt.
  - Feature-Typen: technisch (Qlib Alpha158-Expression-Engine), fundamental (falls Datenquelle vorhanden)
  - Labeling: Triple-Barrier-Methode + Meta-Labeling (mlfinlab), CUSUM-Filter für Event-Sampling, fraktionale Differenzierung für Stationarität

### Definition of Done
- [ ] Mind. 1 Datenquelle liefert bereinigte OHLCV-Daten für das Ziel-Universum
- [ ] Data Quality Gate blockiert nachweislich fehlerhafte Daten (Test mit künstlich verfälschtem Datensatz)
- [ ] Splits/Dividenden werden korrekt in historischen Preisen berücksichtigt (Stichprobe manuell verifiziert)
- [ ] Point-in-Time-Test: Für einen Stichtag t liefert die Pipeline nachweislich keine Daten aus t+1 oder später
- [ ] Triple-Barrier-Labeling implementiert und an einem Testsymbol verifiziert

---

## Phase 2 — Research-Kern & Signal-Baseline

**Ziel:** Feststellen, ob überhaupt ein handelbares Signal existiert — mit der einfachsten robusten Methode.

**Warum vor Backtesting-Infrastruktur:** Ohne ein Kandidaten-Signal gibt es nichts zu backtesten. Und: die Recherche warnt ausdrücklich, dass GBDT (LightGBM) oft die stärkste Baseline ist — das muss zuerst stehen, bevor du Zeit in Deep Learning/Transformer/RL investierst.

### Komponenten
- **Qlib-Setup** als Research-/Feature-Rückgrat (Alpha158-Expression-Engine)
- **LightGBM-Baseline-Modell** auf Alpha158-Features
- **Bewertungsmetriken:** IC, ICIR, Rank IC, Information Ratio
- **Strategy-Artefakt-Konvention:** jede Strategie ab jetzt versioniert mit Metadaten (Universe, Features, Modell-Hash, Hyperparameter) — auch ohne volles MLflow-Setup schon als Konvention (JSON-Metadatei pro Modell-Version), MLflow folgt in Phase 10

### Definition of Done
- [ ] LightGBM-Baseline trainiert auf Alpha158-Features für Ziel-Universum
- [ ] IC/Rank IC/ICIR dokumentiert und nachvollziehbar reproduzierbar (fester Seed, fixe Datensplit-Definition)
- [ ] Modell-Artefakt inkl. Metadaten (Hash, Hyperparameter, Feature-Liste) gespeichert
- [ ] Klares Ja/Nein-Ergebnis: Existiert ein Signal mit IC deutlich über 0? (Kein Go-Live-Grund, aber Grundlage für Phase 3)

---

## Phase 3 — Backtesting & Overfitting-Kontrolle

**Ziel:** Feststellen, ob das Signal aus Phase 2 real ist oder ein Data-Mining-Artefakt.

**Warum vor Portfolio/Risk:** Portfolio-Optimierung auf einem überfitteten Signal zu bauen ist verschwendete Arbeit — die Recherche nennt CPCV/DSR als entscheidendes Kriterium, *bevor* man in Exekution investiert.

### Komponenten
- **vectorbt** für schnelle Signal-Triage (Parameter-Sweeps, "hat das überhaupt Alpha?")
- **NautilusTrader** als event-getriebener Backtester mit Research-zu-Live-Parität (derselbe Code läuft später live — daher hier schon einführen, nicht erst bei Execution)
- **Combinatorial Purged Cross-Validation (CPCV)** mit Purging + Embargoing (mlfinlab)
- **Deflated Sharpe Ratio (DSR)** zur Korrektur von Selection-Bias bei Mehrfachtests
- **Paper-Trading-Modus** in NautilusTrader aktivieren (nächste Vorbereitung für Phase 5)

### Definition of Done
- [ ] Signal-Triage mit vectorbt abgeschlossen, grobe Parameterregion identifiziert
- [ ] Vollständige CPCV-Auswertung des finalen Kandidatensignals durchgeführt
- [ ] Deflated Sharpe Ratio berechnet und **> 0**
- [ ] NautilusTrader-Backtest liefert vergleichbare Ergebnisse wie vectorbt-Triage (Parität-Check, keine großen Abweichungen durch Mikrostruktur-Annahmen)
- [ ] Paper-Trading-Setup lauffähig mit demselben Code wie der Backtest

> **MVP-Zwischen-Gate:** Ohne DSR > 0 hier nicht weitermachen. Zurück zu Phase 2, andere Features/Universum/Zeitraum testen.

---

## Phase 4 — Portfolio-Optimierung & Risk-Engine

**Ziel:** Signale in Positionsgrößen übersetzen und Kapital vor dem System selbst schützen.

**Warum vor Execution:** Ohne Risk-Engine mit Pre-Trade-Checks ist ein Live-System mit echtem Geld fahrlässig — das gilt für Solo-Trader genauso wie für einen Fonds. Diese Phase muss vor jeder echten Order stehen.

### Komponenten
- **Portfolio-Optimierung (Basis): PyPortfolioOpt**
  - Efficient Frontier (Markowitz), Mean-CVaR als Startpunkt
- **Erweiterung: Riskfolio-Lib** für CVaR/CDaR/Ulcer Index, sobald Basis läuft
- **Position Sizing:** fraktionales Kelly-Kriterium, Volatilitäts-Targeting
- **Constraints-Management:** Positions-Limits, Sektor-Limits, Turnover-Limits
- **Dedizierte Risk-Engine mit Pre-Trade-Checks**
  - Jede Order wird vor Absendung gegen Limits validiert (analog NautilusTrader RiskEngine)
  - VaR/CVaR-Berechnung, Drawdown-Kontrolle, Exposure-Limits
- **Kill-Switch/Circuit-Breaker** — manuell und automatisch (z. B. bei X% Tages-Drawdown)

### Definition of Done
- [ ] PyPortfolioOpt liefert Zielgewichte aus Signal-Output
- [ ] Riskfolio-Lib-CVaR-Constraint aktiv und nachweislich limitierend in Stress-Szenario
- [ ] Risk-Engine blockiert nachweislich eine Order, die ein Limit verletzt (Test mit absichtlich verletzender Order)
- [ ] Kill-Switch manuell auslösbar UND automatisch bei definiertem Drawdown-Schwellenwert
- [ ] Alle Limits sind über die Config-Engine aus Phase 0 einstellbar, nicht hartkodiert

---

## Phase 5 — Order-Management, Execution & Paper-Trading

**Ziel:** Orders zuverlässig und ohne unnötige Kosten ausführen.

**Warum jetzt und nicht früher:** Execution-Logik ohne validiertes Signal (Phase 2-3) und ohne Risk-Engine (Phase 4) davor zu bauen, hätte Reihenfolge-Risiken erzeugt (Geld bewegt sich, bevor Sicherheitsnetz steht).

### Komponenten
- **Broker-Adapter** (NautilusTrader-Adapter für Interactive Brokers oder Alpaca — je nach Kontenverfügbarkeit)
- **Execution-Algorithmen (Basis):** TWAP, VWAP — Implementation Shortfall/Almgren-Chriss erst bei Bedarf (Solo-Kapitalgröße macht Market-Impact-Optimierung anfangs meist irrelevant — **kritischer Hinweis:** nicht vorzeitig überkomplizieren)
- **Transaction Cost Analysis (TCA) — minimal:** Slippage-Tracking (Ausführungspreis vs. Benchmark) pro Order
- **Live-Paper-Trading-Phase** (Pflicht-Zwischenschritt, kein Überspringen): mind. mehrere Wochen Paper-Trading mit echten Marktdaten, bevor echtes Kapital fließt

### Definition of Done
- [ ] Broker-Adapter verbindet sich erfolgreich mit Paper-Trading-Konto
- [ ] Order-Lifecycle (Submit → Fill/Partial-Fill/Reject) korrekt geloggt und im Event-Bus sichtbar
- [ ] Slippage wird pro Order getrackt
- [ ] Mind. 4-6 Wochen Paper-Trading ohne kritische Fehler/Abweichungen zum Backtest (definierte Toleranzschwelle für Performance-Abweichung)
- [ ] Live-Parität nachweislich gegeben: gleicher Code-Pfad Backtest ↔ Paper ↔ Live

---

## Phase 6 — Audit, Steuer/Reconciliation & Go-Live-Vorbereitung

**Ziel:** Jede Entscheidung nachvollziehbar machen und die Buchhaltungs-Basis für echtes Geld schaffen.

**Warum vor echtem Go-Live:** Ohne Audit-Trail kannst du im Fehlerfall nicht rekonstruieren, was passiert ist. Ohne Tax-Lot-Tracking hast du ab dem ersten Trade mit echtem Geld ein Steuerproblem, keine Kür.

### Komponenten
- **Unveränderliche, hash-verkettete Audit-Logs** (Inputs, Modellversion, Feature-Werte, Entscheidung — pro Order reproduzierbar)
- **Deterministic Replay** (NautilusTrader durable event log + Replay)
- **Tax-Lot-Accounting** (FIFO/LIFO/HIFO-Tracking) — **Korrektur gegenüber Recherche:** dort als "institutionell/niedrige Priorität" eingestuft, für dich als Privatperson mit echtem Kapital aber ab Tag 1 relevant fürs Finanzamt
- **Reconciliation:** Positions-/Cash-Abgleich zwischen deinem System und dem Broker (mind. täglich automatisiert)
- **Business-Rule-Engine (Ausbau aus Phase 0):** Trading-Limits final konfiguriert, Kill-Switch produktionsscharf

### Definition of Done
- [ ] Jede Order ist rückwirkend vollständig rekonstruierbar (Input-Daten, Modellversion, Entscheidungsbegründung)
- [ ] Hash-Kette der Audit-Logs verifiziert manipulationssicher (Test: Log-Eintrag manipulieren → Kette bricht nachweislich)
- [ ] Tax-Lot-Report für Testzeitraum erzeugt und manuell plausibilisiert
- [ ] Automatischer täglicher Reconciliation-Check zwischen System und Broker-Konto läuft und meldet Abweichungen
- [ ] **Go-Live-Checkliste vollständig abgehakt** (DSR > 0, Risk-Engine aktiv, Paper-Trading-Parität erwiesen, Audit-Trail funktionsfähig, Kill-Switch getestet)

---

## Phase 7 — Monitoring, Analytics & Dashboard

**Ziel:** Operative Sichtbarkeit — merken, wenn etwas schiefläuft, bevor es teuer wird.

**Warum jetzt (nicht früher, nicht später):** Vor Go-Live nice-to-have, ab Go-Live Pflicht. Baut auf allen vorherigen Phasen auf (braucht Daten aus Execution, Risk, Audit).

### Komponenten
- **Performance-Attribution:** Sharpe, Sortino, Calmar, Max Drawdown, IC/ICIR (pyfolio/quantstats/empyrical)
- **Dashboard (Solo-Version):** Streamlit oder Plotly Dash statt vollem React+TypeScript-Setup — Begründung: geringerer Wartungsaufwand für Einzelperson, Grafana als Alternative falls bereits vorhanden
- **KPIs/Views:** Equity-Kurve, aktuelle Positionen/Exposures, P&L, Risikometriken, TCA/Slippage-Reports, Order-Blotter
- **Live-vs-Backtest-Drift-Monitoring** (zentrale Warnfunktion — meldet, wenn Live-Performance signifikant vom Backtest abweicht)

### Definition of Done
- [ ] Dashboard zeigt Equity-Kurve, offene Positionen, aktuelle Risikometriken in Echtzeit/nahezu Echtzeit
- [ ] Performance-Kennzahlen (Sharpe, Sortino, Calmar) werden automatisch berechnet und historisch gespeichert
- [ ] Drift-Alert löst nachweislich aus, wenn Live-Performance eine definierte Abweichungsschwelle zum Backtest überschreitet

---

# — MVP-GATE —
**Kriterium für den Übergang zu Advanced-Phasen:** Deflated Sharpe Ratio > 0 unter CPCV, nachgewiesene Paper-Trading-Parität, funktionierender Audit-Trail, Risk-Engine live getestet. Ohne dieses Gate keine der folgenden Phasen — sie sind Ausbau, kein Ersatz für ein fehlendes Fundament.

---

## Phase 8 — FX-Erweiterung

**Ziel:** Zweite Assetklasse sauber nachziehen, statt sie von Anfang an parallel zu bauen.

**Warum erst hier:** FX hat andere Handelszeiten, Margin-/Hebel-Mechanik, Swap-/Rollover-Kosten und andere Datenquellen als Aktien. Parallel zum Aktien-MVP zu bauen hätte Komplexität ohne Not addiert — ein Signal muss auch pro Assetklasse einzeln validiert werden (Phase 2-3 gelten pro Assetklasse).

### Komponenten
- **FX-Daten-Adapter** (separate Vendor-Integration, andere Zeitzonen-/Handelszeiten-Logik)
- **FX-spezifische Risikoparameter** (Hebel, Margin-Anforderungen, Swap-/Rollover-Kosten in Risk-Engine integrieren)
- **FX-Signal-Validierung** (eigene CPCV/DSR-Auswertung — Aktien-Ergebnis überträgt sich nicht automatisch)
- **FX-Broker-Adapter/Execution** (ggf. separater Broker als Aktien-Broker)

### Definition of Done
- [ ] FX-Datenpipeline liefert saubere, point-in-time-korrekte Daten
- [ ] Eigenständiger DSR > 0 für FX-Signal nachgewiesen (nicht vom Aktien-Signal übernommen)
- [ ] Risk-Engine berücksichtigt FX-Hebel/Margin korrekt
- [ ] FX-Paper-Trading-Phase analog Phase 5 durchlaufen

---

## Phase 9 — Explainability & Multi-Agent-Layer *(Advanced/Optional)*

**Ziel:** Verstehen, warum das System handelt — als Debugging-/Vertrauens-Werkzeug, **nicht** als Compliance-Pflicht (die gilt für dich als Privatperson nicht in der SR-11-7-Form).

**Kritische Einordnung:** Die Recherche behandelt LangGraph-Multi-Agent-Orchestrierung mit Investoren-Personas als "HOCH"-Priorität. Für dich als Solo-Trader mit eigenem Kapital ist das explizit **optional** — es strukturiert Entscheidungen und kann Erklärbarkeit verbessern, ersetzt aber keine quantitative Validierung. Baue es nur, wenn Phase 0-8 stabil laufen und du zusätzliche Reasoning-Schicht über bereits validierten Signalen willst — nicht als Ersatz für Signalqualität.

### Komponenten
- **SHAP-Explainability** auf dem Forecasting-Modell (globale + lokale Attributionen, Feature-Drift-Erkennung)
- **LangGraph-Multi-Agent-Orchestrierung** (falls gewünscht): Research-/Sentiment-/Risk-Agenten als zusätzliche Reasoning-Schicht, Portfolio-Manager-Agent synthetisiert
- **Kommunikationsprotokoll:** gemeinsamer LangGraph-AgentState, ggf. MCP für externe Tools

### Definition of Done
- [ ] SHAP-Analyse läuft periodisch gegen Live-Outputs und erkennt Feature-Drift in Testfall
- [ ] (Falls umgesetzt) Multi-Agent-Pipeline liefert nachvollziehbare Entscheidungsprotokolle, die in Audit-Log (Phase 6) einfließen
- [ ] Klar dokumentiert: Agenten-Layer ist additiv, finale Risk-/Order-Entscheidung bleibt bei der deterministischen Risk-Engine aus Phase 4

---

## Phase 10 — Strategy-Registry, Ensemble & Workflow-Orchestrierung *(Advanced)*

**Ziel:** Mehrere Strategien/Signale sauber verwalten, wenn eine einzelne Strategie nicht mehr reicht.

**Warum erst hier:** Ensemble und Registry lohnen sich erst, wenn mehr als eine validierte Strategie existiert (Phase 2-3, ggf. mehrfach durchlaufen). Vorher ist MLflow/Prefect/Dagster ungenutzte Infrastruktur.

### Komponenten
- **MLflow Model Registry** (Staging/Production-Promotion, Experiment-Tracking) — löst die einfache JSON-Metadaten-Konvention aus Phase 2 ab
- **Optuna** für Hyperparameter-Suche
- **Signal-Ensemble:** gewichtete Kombination mehrerer validierter Alpha-Quellen, Meta-Labeling zur Bet-Size-Bestimmung
- **Feast-Feature-Store** (Nachrüstung, falls die eigene Pipeline aus Phase 1 an Grenzen stößt — Point-in-Time-Konzept bleibt gleich, nur die Plattform wechselt)
- **Workflow-Orchestrierung:** Prefect oder Dagster für Pipeline-Scheduling — Solo-Einstieg über einfaches Cron/Skript-Scheduling ist bis hierhin ausreichend, Wechsel erst bei echtem Multi-Pipeline-Bedarf

### Definition of Done
- [ ] Mind. 2 unabhängig validierte Signale im Ensemble kombiniert, Ensemble schlägt Einzelsignale risikoadjustiert im Backtest
- [ ] MLflow-Registry verwaltet Modell-Versionen mit Staging/Production-Status
- [ ] Workflow-Orchestrator übernimmt Scheduling von Datenpipeline + Retraining-Jobs

---

## Phase 11 — Marktregime & Sentiment *(Advanced)*

**Ziel:** Adaptive Strategiewahl je nach Marktphase, zusätzliche alternative Datenquelle.

**Warum erst hier:** Beides sind Verfeinerungen einer bereits funktionierenden Basis — Regime-Erkennung ohne validierte Basisstrategie hat nichts, was es rotieren könnte.

### Komponenten
- **Hidden Markov Models (hmmlearn)** zur Regime-Erkennung (Bull/Bear/Seitwärts, High-/Low-Vol)
- **Regime-abhängige Strategierotation:** Trend-Following in Bull-Regimes, Mean-Reversion seitwärts, Risikoreduktion in High-Vol
- **FinBERT-Sentiment-Agent** als zusätzliche Feature-Quelle

### Definition of Done
- [ ] HMM erkennt nachweislich unterschiedliche Regimes im historischen Testzeitraum (manuell plausibilisiert gegen bekannte Marktphasen)
- [ ] Regime-abhängige Rotation verbessert risikoadjustierte Kennzahlen gegenüber statischer Strategie im Backtest
- [ ] FinBERT-Feature zeigt messbaren (wenn auch kleinen) IC-Beitrag

---

## Phase 12 — Selbstoptimierung, Alpha-Decay & Monitoring *(Advanced)*

**Ziel:** Sicherstellen, dass die Strategie über Zeit nicht verfällt.

**Warum erst hier:** Setzt eine laufende Live-Strategie mit Historie voraus — ohne Live-Daten gibt es nichts, dessen Decay man messen könnte.

### Komponenten
- **Walk-Forward-Retraining** (z. B. 3-Monats-Rolling, gegen Concept-Drift)
- **Automatisches Retraining-Pipeline** (Optuna + Ray Tune, falls Rechenkapazität vorhanden — sonst Optuna allein ausreichend)
- **Alpha-Decay-Monitoring:** kontinuierliche IC-Überwachung, Live-vs-Backtest-Drift
- **Capacity-Hinweis:** Bei eigenem Kapital in überschaubarer Größenordnung ist Capacity-Management (Recherche-Abschnitt "Weitere Module") **nicht relevant** — erst bei deutlich wachsendem AUM prüfen

### Definition of Done
- [ ] Retraining-Pipeline läuft automatisiert nach definiertem Intervall
- [ ] Alpha-Decay-Dashboard zeigt IC-Verlauf über Zeit
- [ ] Klare Schwelle definiert, ab der eine Strategie automatisch pausiert wird (statt manuellem Eingreifen)

---

## Phase 13 — RL & Transformer-Forschungsspur *(Optional/Research)*

**Ziel:** Prüfen, ob exotischere Modelle die GBDT-Baseline tatsächlich schlagen — nicht, weil sie neu sind.

**Warum zuletzt:** Die Recherche selbst stuft dies als "niedrige Priorität" ein: hoher Aufwand, marginaler nachgewiesener Gewinn, live unbewiesen. Ohne stabile Basis (Phase 0-8) ist das reine Zeitverschwendung.

### Komponenten
- **FinRL-Ensemble** (PPO/A2C/DDPG) — nur falls Baseline-Vergleich das rechtfertigt
- **Transformer-Forecasting** (PatchTST, TFT) — nur gegen GBDT-Baseline validiert, nicht als Ersatz
- **RD-Agent-artige autonome Research-Loop** — explizit als Experiment markiert, **keine** Produktionsabhängigkeit

### Definition of Done
- [ ] Jedes exotische Modell hat einen dokumentierten A/B-Vergleich gegen die LightGBM-Baseline auf denselben CPCV-Splits
- [ ] Nur bei nachgewiesener, risikoadjustierter Verbesserung wird ein Modell in die Signal-Engine (Phase 10) aufgenommen
- [ ] Kein exotisches Modell ersetzt die Baseline, ohne dass sie parallel weiterläuft (Fallback)

---

## Anhang — Institutionelle Erweiterung (NICHT Teil dieser Roadmap)

Falls du später Fremdkapital aufnimmst, werden folgende Recherche-Komponenten relevant, die hier bewusst ausgeklammert sind:
- Volle Compliance/Regulatory-Reporting-Engine (Best-Execution-Nachweis für Dritte, Restricted Lists für mehrere Mandate)
- Capacity-Management (AUM-Skalierungs-Map)
- Formale SR-11-7-Dokumentation als regulatorische Pflicht (bei dir aktuell freiwilliges Debugging-Tool, siehe Phase 9)
- Multi-Tenant-/Mandats-Trennung

---

## Modul-Abhängigkeiten (Übersicht)

```
Phase 0 (Architektur/Message-Bus/Config) ─┬─ Grundlage für ALLE folgenden Phasen (cross-cutting)
                                            │
Phase 1 (Daten/Feature-Pipeline) ──────────┼─→ Phase 2 (Research/Signal)
                                            │
Phase 2 (Signal-Baseline) ─────────────────┼─→ Phase 3 (Backtest/CPCV/DSR)
                                            │
Phase 3 (Backtest bestanden) ──────────────┼─→ Phase 4 (Portfolio/Risk-Engine)
                                            │
Phase 4 (Risk-Engine steht) ───────────────┼─→ Phase 5 (Order-Management/Execution)
                                            │
Phase 5 (Paper-Trading-Parität) ───────────┼─→ Phase 6 (Audit/Tax/Reconciliation)
                                            │
Phase 6 (Audit-Trail fertig) ──────────────┼─→ Phase 7 (Monitoring/Dashboard)
                                            │
                                    [MVP-GATE: DSR>0 + Parität + Audit + Risk-Engine]
                                            │
                                            ├─→ Phase 8 (FX) — braucht Phase 1,4,5 pro Assetklasse neu
                                            ├─→ Phase 9 (Explainability/Agents) — braucht Phase 2,4,6
                                            ├─→ Phase 10 (Registry/Ensemble/Workflow) — braucht ≥2× Phase 2-3 durchlaufen
                                            │        │
                                            │        └─→ Phase 12 (Selbstoptimierung) — braucht Phase 10 (Registry) + Live-Historie aus Phase 5-7
                                            ├─→ Phase 11 (Regime/Sentiment) — braucht Phase 1 (Features) + Phase 7 (Monitoring zur Validierung)
                                            └─→ Phase 13 (RL/Transformer) — braucht Phase 2 (Baseline zum Schlagen) + Phase 3 (CPCV-Harness)
```

**Kernregel:** Phase 0-7 sind strikt sequenziell (jede baut auf der vorherigen auf, kein Überspringen). Phase 8-13 sind untereinander weitgehend unabhängig und können nach Priorität/Interesse in beliebiger Reihenfolge angegangen werden — mit den oben genannten Abhängigkeiten zu den MVP-Phasen.

---

## Fortschritts-Checkliste (Master)

- [ ] **Phase 0** — Architektur- & Projekt-Fundament
- [ ] **Phase 1** — Daten-Fundament
- [ ] **Phase 2** — Research-Kern & Signal-Baseline
- [ ] **Phase 3** — Backtesting & Overfitting-Kontrolle (DSR > 0 erreicht)
- [ ] **Phase 4** — Portfolio-Optimierung & Risk-Engine
- [ ] **Phase 5** — Order-Management, Execution & Paper-Trading
- [ ] **Phase 6** — Audit, Steuer/Reconciliation & Go-Live-Vorbereitung
- [ ] **Phase 7** — Monitoring, Analytics & Dashboard
- [ ] **— MVP-GATE erreicht —**
- [ ] **Phase 8** — FX-Erweiterung
- [ ] **Phase 9** — Explainability & Multi-Agent-Layer (optional)
- [ ] **Phase 10** — Strategy-Registry, Ensemble & Workflow-Orchestrierung
- [ ] **Phase 11** — Marktregime & Sentiment
- [ ] **Phase 12** — Selbstoptimierung & Alpha-Decay-Monitoring
- [ ] **Phase 13** — RL & Transformer-Forschungsspur (optional/research)

---

## Offene Risiken (aus der Recherche übernommen, im Auge behalten)

- **Riskfolio-Lib** wird von einem einzelnen Hauptentwickler gepflegt — Abhängigkeitsrisiko, ggf. Fork/Vendoring erwägen, sobald produktiv genutzt.
- **Backtrader** ist Maintenance-Modus — hier bewusst nicht verwendet (NautilusTrader/vectorbt stattdessen).
- **LangGraph/CrewAI/AutoGen** ändern sich schnell — falls Phase 9 umgesetzt wird, Versionen strikt pinnen.
- **RD-Agent(Q)-Zahlen** (Phase 13, falls überhaupt verfolgt) sind Autoren-Backtests auf chinesischen Indizes mit spezifischem LLM-Backend — keine Live-Ergebnisse, nicht als Zielgröße verwenden.
- **Alpha-Decay ist real:** Je öffentlicher eine Methode (Open-Source-Repos), desto schneller wird sie wegarbitragiert — nachhaltiger Vorteil liegt in Datenqualität und Ausführung, nicht im Klonen von Referenz-Repos.
