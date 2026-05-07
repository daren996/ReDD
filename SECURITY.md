# Security Policy

## Supported Versions

ReDD is currently pre-1.0 research software. Security fixes target the current
`main` branch unless a release branch is explicitly announced.

## Reporting A Vulnerability

Do not open public issues for suspected credential leaks, prompt-injection
exposures, unsafe file handling, or dependency vulnerabilities.

Instead, contact the maintainers privately through the repository owner profile
or the security contact configured on GitHub. Include:

- affected commit or release,
- reproduction steps,
- affected provider or dataset configuration,
- whether any credentials, private documents, or generated artifacts may have
  been exposed.

## Secrets And Local Data

- Do not commit `.env`, `api_keys.json`, provider keys, or generated credentials.
- Prefer environment variables referenced by `api_key_env` in config files.
- Treat documents, parquet files, embeddings, logs, and LLM outputs as potentially
  sensitive unless the dataset is explicitly public.
- Generated runtime artifacts belong under ignored paths such as `outputs/`,
  `logs/`, or local dataset cache files.

## Provider And Model Use

LLM providers may receive document text and prompts during execution. Review the
provider's data handling policy before running ReDD on private documents.
