# Security Policy

## Supported versions

Security fixes are applied to the latest release and the `main` branch.

## Reporting a vulnerability

Do not open a public issue for vulnerabilities or leaked credentials. Use the
repository's private GitHub security advisory flow and include affected versions,
reproduction steps, impact, and suggested mitigation when available.

## Operational requirements

- Never commit `.env`, API keys, production telemetry, customer manuals, or database snapshots.
- Production API authentication must be enabled with secrets supplied by the deployment platform.
- All maintenance actions require authenticated human approval and an audit event.
- Model and RAG artifacts must be checksum-verified before loading.
- Rotate any credential that appears in logs, screenshots, chat, or Git history.
