# setup_env.ps1
# This script sets up a Python virtual environment and installs dependencies.

$ErrorActionPreference = "Stop"

Write-Host "--- Garmin Mosaic Setup Script ---" -ForegroundColor Cyan

try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Python found: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "[ERROR] Python is not installed or not in your PATH." -ForegroundColor Red
    exit 1
}

Write-Host "Creating virtual environment in .venv..." -ForegroundColor Yellow
python -m venv .venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
    exit 1
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] Virtual environment Python not found at $venvPython" -ForegroundColor Red
    exit 1
}

Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to upgrade pip." -ForegroundColor Red
    exit 1
}

& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Dependency installation failed." -ForegroundColor Red
    exit 1
}

Write-Host "----------------------------------------------------" -ForegroundColor Green
Write-Host "[OK] Setup complete!" -ForegroundColor Green
Write-Host "Virtual environment: .\.venv" -ForegroundColor White
Write-Host "Mosaic script: .\garmin_mosaic.py" -ForegroundColor White
Write-Host "----------------------------------------------------" -ForegroundColor Green
