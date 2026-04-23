## Summary

- Explain the user-facing or package-facing change.

## Validation

- [ ] `python -m unittest discover -s tests -v`
- [ ] `ruff check src/redd/__init__.py src/redd/api.py src/redd/config.py src/redd/runtime.py src/redd/core/data_population/factory.py src/redd/core/schema_gen/factory.py src/redd/core/utils/prompt_utils.py src/redd/core/llm/providers.py tests`
- [ ] `mypy`
- [ ] `python -m build`

## Boundary Check

- [ ] Public API changes were documented in `README.md` when needed
- [ ] Newly migrated modules were classified in `docs/MODULE_CLASSIFICATION.md`
- [ ] External integrations remain behind adapters or factories
