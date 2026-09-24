<#
.SYNOPSIS
  Start the GenieX model server on the NPU (if it isn't running) and then Netra.
#>
param([switch]$NoHotkeys)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

function Test-Brain {
    try { Invoke-RestMethod "http://127.0.0.1:18181/v1/models" -TimeoutSec 2 | Out-Null; return $true }
    catch { return $false }
}

if (-not (Test-Brain)) {
    Write-Host "Starting GenieX server on the Hexagon NPU..." -ForegroundColor Cyan
    Start-Process -WindowStyle Minimized geniex -ArgumentList "serve"
    foreach ($i in 1..60) { if (Test-Brain) { break }; Start-Sleep -Seconds 1 }
    if (-not (Test-Brain)) { throw "GenieX server did not start on port 18181. Run 'geniex serve' manually to see why." }
}

Push-Location $Root
try {
    $netraArgs = @("run")
    if ($NoHotkeys) { $netraArgs += "--no-hotkeys" }
    uv run netra @netraArgs
} finally { Pop-Location }
