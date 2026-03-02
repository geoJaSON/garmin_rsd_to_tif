# setup_env.ps1
# This script sets up a Python virtual environment and installs dependencies for garmin_mosaic.py

Write-Host "--- Garmin Mosaic Setup Script ---" -ForegroundColor Cyan

# 1. Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Error: Python is not installed or not in your PATH." -ForegroundColor Red
    exit 1
}

# 2. Create Virtual Environment
Write-Host "Creating virtual environment in .venv..." -ForegroundColor Yellow
python -m venv .venv

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to create virtual environment." -ForegroundColor Red
    exit 1
}

# 3. Upgrade pip and Install Dependencies
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "----------------------------------------------------" -ForegroundColor Green
    Write-Host "✓ Setup complete!" -ForegroundColor Green
    Write-Host "To use the environment, run:" -ForegroundColor White
    Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To run the mosaic script:" -ForegroundColor White
    Write-Host "  python Misc/garmin_mosaic.py" -ForegroundColor Yellow
    Write-Host "----------------------------------------------------" -ForegroundColor Green
} else {
    Write-Host "✗ An error occurred during dependency installation." -ForegroundColor Red
}
