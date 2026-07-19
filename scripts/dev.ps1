$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "Starting MechSage backend and frontend..."
Start-Process -FilePath python -ArgumentList @("-m", "uvicorn", "backend.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000") -WorkingDirectory $repoRoot -WindowStyle Hidden
Start-Process -FilePath npm.cmd -ArgumentList @("run", "dev") -WorkingDirectory (Join-Path $repoRoot "frontend") -WindowStyle Hidden

Write-Host "Frontend: http://localhost:3000"
Write-Host "Backend:  http://localhost:8000/docs"
