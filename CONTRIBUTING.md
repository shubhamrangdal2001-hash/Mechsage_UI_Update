# Contributing to MechSage

## Development workflow

1. Create a branch using `feature/`, `fix/`, `docs/`, or `chore/`.
2. Copy `.env.example` to `.env`; never commit credentials.
3. Keep changes focused and add tests for changed behavior.
4. Run `scripts/verify.ps1` before opening a pull request.
5. Complete the pull-request template, including ML/RAG impact.

## Local services

```powershell
docker compose up --build
```

Or run the API and web application separately as documented in
[`docs/operations/local-development.md`](docs/operations/local-development.md).

## Commit and review rules

- Use imperative commit subjects, such as `fix: deduplicate live work orders`.
- Do not commit databases, logs, raw datasets, vector indexes, or secrets.
- Model changes must include artifact checksums, evaluation evidence, and feature-schema compatibility.
- Corpus changes must include retrieval evaluation and citation checks.
- Maintenance actions must continue to require human approval.

## Pull-request requirements

- CI must pass.
- At least one code-owner review is required.
- User-visible behavior requires API evidence, screenshots, or an E2E test.
- Breaking API, model, or corpus changes require release notes.
