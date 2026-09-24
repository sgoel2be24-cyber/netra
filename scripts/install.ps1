<#
.SYNOPSIS
  One-shot setup for Netra on a Snapdragon X / X2 Windows laptop.

.DESCRIPTION
  1. Installs uv and a *native ARM64* Python 3.12 (x64 Python under emulation cannot reach the NPU).
  2. Installs Netra's dependencies, including onnxruntime-qnn (Hexagon NPU execution provider).
  3. Downloads Qualcomm AI Hub's precompiled Whisper encoder/decoder for your NPU generation.
  4. Installs Qualcomm GenieX and pulls the vision-language model that runs on the NPU.

.EXAMPLE
  Set-ExecutionPolicy -Scope Process Bypass -Force
  .\scripts\install.ps1
#>
param(
    [string]$BrainModel = "google/gemma-4-E2B-it-qat-q4_0-gguf",
    [switch]$SkipBrain
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is 10x faster without the progress bar
$Root = Split-Path -Parent $PSScriptRoot
$Cache = Join-Path $HOME ".cache\netra"
$PyArm = "cpython-3.12-windows-aarch64-none"

function Step([string]$msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User") + ";$HOME\.local\bin"
}

Step "Checking hardware"
$cpu = (Get-ItemProperty "HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0").ProcessorNameString
$arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
Write-Host "CPU: $cpu ($arch)"
$isSnapdragon = ($arch -eq "ARM64") -and ($cpu -match "Snapdragon|Qualcomm")
if (-not $isSnapdragon) {
    Write-Warning "This is not a Snapdragon PC. Netra will install, but only CPU fallbacks will work."
}
# X2 Elite / X2 Plus use a newer Hexagon NPU and need their own AI Hub assets.
$chipset = if ($cpu -match "X2") { "qualcomm-snapdragon-x2-elite" } else { "qualcomm-snapdragon-x-elite" }
Write-Host "AI Hub chipset: $chipset"

Step "Installing uv"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    Refresh-Path
}

Step "Installing native ARM64 Python 3.12 and Netra's dependencies"
uv python install $PyArm
Push-Location $Root
try {
    uv sync --python $PyArm --extra camera
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Camera extra failed to install on this machine; continuing without webcam support."
        uv sync --python $PyArm
    }
    uv run netra doctor
} finally { Pop-Location }

Step "Downloading Whisper for the Hexagon NPU (Qualcomm AI Hub, $chipset)"
$whisperDir = Join-Path $Cache "whisper_npu"
if (-not (Test-Path (Join-Path $whisperDir "encoder.onnx"))) {
    $tmp = Join-Path $env:TEMP "netra-whisper"
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    uvx --python $PyArm qai-hub-apps fetch whisper_windows_py --model whisper_base --chipset $chipset --output-dir $tmp
    $models = Get-ChildItem $tmp -Recurse -Filter "encoder.onnx" | Select-Object -First 1
    if (-not $models) { throw "qai-hub-apps did not produce encoder.onnx; see https://aihub.qualcomm.com/models/whisper_base" }
    New-Item -ItemType Directory -Force $whisperDir | Out-Null
    Copy-Item (Join-Path $models.DirectoryName "*") $whisperDir -Recurse -Force
}
Write-Host "Whisper NPU model: $whisperDir"

if (-not $SkipBrain) {
    Step "Installing Qualcomm GenieX (runs the vision-language model on the NPU)"
    Refresh-Path
    if (-not (Get-Command geniex -ErrorAction SilentlyContinue)) {
        $exe = Join-Path $env:TEMP "geniex-cli.exe"
        Invoke-WebRequest "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/geniex-cli.exe" -OutFile $exe
        Write-Host "The GenieX installer will open. It is not code-signed yet: if SmartScreen warns, choose 'More info' then 'Run anyway'."
        Start-Process $exe -Wait
        Refresh-Path
    }
    Step "Pulling $BrainModel (first run downloads about 4 GB)"
    geniex pull $BrainModel --model-type vlm
}

Step "Done"
Write-Host "Start Netra with:  .\scripts\start.ps1" -ForegroundColor Green
