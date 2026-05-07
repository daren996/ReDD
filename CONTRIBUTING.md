# Contributing

## Setup

1. Install Python 3.10 or newer.
2. Install the package and development tooling:

   ```bash
   python -m pip install -e ".[dev]"
   ```

3. Provide provider API keys only through environment variables or local untracked files such as `api_keys.json`.

## Validation

Run the local validation suite before opening a pull request:

```bash
pytest -q
ruff check src/redd/__init__.py src/redd/api.py src/redd/config.py src/redd/runtime.py src/redd/core/data_population/factory.py src/redd/core/schema_gen/factory.py src/redd/core/utils/prompt_utils.py src/redd/llm/providers.py tests
mypy
python -m build
```

## Packaging And Boundaries

- Import `redd` from installed package surfaces instead of mutating `sys.path`.
- Keep command-line wrappers in `scripts/` thin; CLI argument parsing belongs in `src/redd/cli/`.
- Treat `src/redd/*.py` as the public package contract unless a module is explicitly documented as internal-only.
- Keep external service integrations behind adapters or factories so providers can be swapped without leaking implementation details across the package.

## Migration Policy

- Treat `src/redd/exp/`, `src/redd/correction/`, and external research branches as incubators for future migrations.
- Only promote modules that have a documented config surface, an on/off switch or fallback path, and tests.
- For every migrated area, update `docs/MODULE_CLASSIFICATION.md` so the public/internal/experiment-only split stays explicit.
- Prefer adding regression tests around package entry points before moving more experimental code into public runtime paths.
