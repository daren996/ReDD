# Test Layout

This project keeps tests centralized under `tests/`.

- `unit/`: focused tests for individual modules and small behaviors.
- `contracts/`: public API and backward-compatibility expectations.
- `integration/`: multi-module flows with external services mocked.
- `smoke/`: import, package, and CLI wiring checks.

Pytest is configured in `pyproject.toml` to collect from `tests/` only.
