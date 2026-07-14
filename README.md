# AI Hedge Fund

Phase 0 provides the vendor-neutral foundation for a Python 3.12, event-driven
trading platform. Module boundaries use immutable Pydantic DTOs; trading rules
and the instrument universe come from validated YAML configuration.

## Architecture

The repository follows a hexagonal (ports-and-adapters) layout:

```text
src/aihedgefund/
├── core/       # DTOs, message-bus port/implementation, config models, ports
├── config/     # Packaged YAML configuration
├── data/       # Market-data domain
├── research/   # Research domain
├── signals/    # Signal domain
├── portfolio/  # Portfolio domain
├── risk/       # Risk domain
├── execution/  # Execution domain
├── backtest/   # Backtesting domain
├── audit/      # Audit domain
└── dashboard/  # Dashboard domain
```

`core/` and the domain packages contain no vendor imports. Concrete integrations
will be added in an `adapters/` subpackage of the owning domain in later phases.
They must implement the abstract contracts in `core/ports.py`.

Commands express intent and are dispatched separately from events, which record
facts. `Order` is a command; `Signal`, `Fill`, `RiskCheck`, and `FeatureVector`
are events. `InProcessMessageBus` dispatches synchronously in subscriber
registration order so execution is deterministic and can later support replay.

## Configuration

`src/aihedgefund/config/limits.yaml` is the single Phase 0 source for trading
limits, the universe, and feature flags. Consumers call `load_settings()` and
receive a validated `Settings` model. Missing, malformed, or invalid
configuration fails at load time; raw YAML mappings are not exposed globally.

## Phase 1: data foundation

Phase 1 adds a hexagonal market-data foundation without changing the Phase 0
message-bus or trading DTO contracts:

- `data/` owns the provider port, retry/fallback service, yfinance adapter,
  corporate-action transforms, and hard-fail quality gate.
- `features/` computes hand-rolled causal technical features and provides
  point-in-time join/look-ahead guards. Feature matrices use a sorted
  `(timestamp, symbol)` index.
- `labels/` provides causal volatility and event sampling, triple-barrier
  directional/meta labels, overlap-aware sample weights, and fixed-width
  fractional differentiation. Labels carry both event start (`t0`) and
  barrier-touch (`t1`) timestamps for future purged cross-validation.

The packaged YAML adds `start`, `end`, `frequency`, `symbol_aliases`, `quality`,
`labels`, and `fracdiff` settings. All are validated by the existing
`load_settings()` entry point; invalid or missing values fail at load time.

Unit tests are entirely offline and use deterministic synthetic data. For an
explicit live smoke test only, run the non-pytest script:

```bash
poetry run python scripts/manual_yfinance_check.py
```

## Development

Install the exact locked dependencies and run the same checks as CI:

```bash
poetry install
poetry run ruff check .
poetry run mypy src/
poetry run pytest -q
```