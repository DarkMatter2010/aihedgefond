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

## Development

Install the exact locked dependencies and run the same checks as CI:

```bash
poetry install
poetry run ruff check .
poetry run mypy src/
poetry run pytest -q
```