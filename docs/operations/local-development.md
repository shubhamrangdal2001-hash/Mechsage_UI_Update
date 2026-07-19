# Local development

## Prerequisites

- Python 3.11
- Node.js 20
- Docker Desktop when using Compose

## Configuration

Copy `.env.example` to `.env` and replace placeholder values. Do not commit `.env`.

## Native development

```powershell
python -m pip install -r backend\requirements.txt
Push-Location frontend
npm.cmd ci
Pop-Location
.\scripts\dev.ps1
```

The backend may take approximately one minute to load model artifacts and fit
feature scalers. Readiness is available from `GET /health`.

## Containers

```powershell
docker compose up --build
```

## Verification

```powershell
.\scripts\verify.ps1
```
