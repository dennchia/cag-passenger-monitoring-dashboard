# start.ps1 - Located at the root of your project
Write-Host "Launching CAG Passenger Monitoring Dashboard V1..." -ForegroundColor Cyan

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"
$BackendPython = Join-Path $Backend ".venv\Scripts\python.exe"
$Npm = "C:\Program Files\nodejs\npm.cmd"

if (-not (Test-Path $BackendPython)) {
    Write-Host "Backend virtual environment not found. Run the backend setup first." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $Npm)) {
    $Npm = "npm.cmd"
}

# 1. Start Backend in a new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Backend'; & '$BackendPython' -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

# 2. Wait 2 seconds for backend to initialize
Start-Sleep -Seconds 2

# 3. Start Frontend in a new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Frontend'; & '$Npm' run dev -- --host 0.0.0.0"

Write-Host "Both servers launching. Check http://localhost:5173 in your browser." -ForegroundColor Green
