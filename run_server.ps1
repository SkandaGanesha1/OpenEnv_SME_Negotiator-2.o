$ErrorActionPreference = 'Stop'

$port = 7860
$projectRoot = $PSScriptRoot

Write-Host "Checking for existing process on port $port..."
$existing = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue

if ($existing) {
    $ownerPids = $existing | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($ownerPid in $ownerPids) {
        Write-Host "Stopping process $ownerPid using port $port..."
        Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 500
}

Write-Host "Starting OpenEnv server on http://127.0.0.1:$port ..."
Set-Location $projectRoot
python -m uvicorn server.app:app --host 127.0.0.1 --port $port
