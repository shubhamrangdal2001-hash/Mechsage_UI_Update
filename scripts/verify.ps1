$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

Push-Location $repoRoot
try {
    python -m compileall -q backend inference dev\agentic
    python -m pytest backend\tests -q
    Push-Location frontend
    try {
        npm.cmd run lint
        node node_modules\typescript\bin\tsc --noEmit
        npm.cmd run build
    }
    finally {
        Pop-Location
    }
}
finally {
    Pop-Location
}
