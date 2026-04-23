# Open Source Readiness

This checklist captures the publishability assumptions that are now encoded in the repository.

## Audit Outcome

- No tracked `api_keys.json` file is present in the repository root.
- Provider credentials are expected through environment variables or local untracked files.
- `.gitignore` now covers local virtualenvs, local key files, and common type/lint caches.
- Prompt loading supports packaged resources, so prompt resolution no longer depends on the current working directory.

## Local Assumptions That Still Exist

- Datasets in `dataset/` are treated as local workspace assets and are not packaged into the wheel.
- Research artifacts in `papers/` are repository assets, not runtime package data.
- Cloud-backed runs still require provider credentials at execution time.
- Some advanced paths under `src/redd/correction/` and `src/redd/exp/` remain research-oriented and are not presented as stable package API.

## Release Baseline

The repository now includes:

- installable package metadata in `pyproject.toml`
- a development extra for build, lint, and type-check tooling
- CI validation for editable install, build, tests, lint, and type-check
- contribution and repository template files under `.github/`

## Ongoing Expectations

- Keep new external integrations behind factories or adapters.
- Add tests before promoting additional `ReDD_Dev` modules into stable package surfaces.
- Re-run the validation commands from `CONTRIBUTING.md` before release tagging or major refactors.
