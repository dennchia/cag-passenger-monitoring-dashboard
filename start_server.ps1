# start_server.ps1 - Centralised server mode for CAG Passenger Monitoring Dashboard
Write-Host "Launching CAG Passenger Monitoring Dashboard - Server Mode..." -ForegroundColor Cyan

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
    $NpmCommand = Get-Command "npm.cmd" -ErrorAction SilentlyContinue
    if ($NpmCommand) {
        $Npm = $NpmCommand.Source
    } else {
        Write-Host "Node.js/npm was not found. Install Node LTS v20+ before running server mode." -ForegroundColor Red
        exit 1
    }
}

try {
    $mqttListener = Get-NetTCPConnection -State Listen -LocalPort 1883 -ErrorAction SilentlyContinue
} catch {
    $mqttListener = $null
}

if ($mqttListener) {
    Write-Host "MQTT broker detected on port 1883." -ForegroundColor Green
} else {
    Write-Host "MQTT broker was not detected on port 1883." -ForegroundColor Yellow
    Write-Host "Start Mosquitto before expecting CV telemetry, or keep MQTT_ENABLED=false for UI-only testing." -ForegroundColor Yellow
}

Write-Host "Building React frontend..." -ForegroundColor Cyan
Push-Location $Frontend
$PreviousViteApiUrl = $env:VITE_API_URL
$env:VITE_API_URL = " "
& $Npm run build
$BuildExitCode = $LASTEXITCODE
if ([string]::IsNullOrEmpty($PreviousViteApiUrl)) {
    Remove-Item Env:\VITE_API_URL -ErrorAction SilentlyContinue
} else {
    $env:VITE_API_URL = $PreviousViteApiUrl
}
if ($BuildExitCode -ne 0) {
    Pop-Location
    Write-Host "Frontend build failed. Fix the frontend build before starting server mode." -ForegroundColor Red
    exit $BuildExitCode
}
Pop-Location

$lanIps = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object {
        $_.IPAddress -notlike "127.*" -and
        $_.IPAddress -notlike "169.254.*" -and
        $_.PrefixOrigin -ne "WellKnown"
    } |
    Select-Object -ExpandProperty IPAddress -Unique

Write-Host ""
Write-Host "Dashboard URLs:" -ForegroundColor Green
Write-Host "  Local:   http://localhost:8000" -ForegroundColor Green
foreach ($ip in $lanIps) {
    Write-Host "  Network: http://$ip`:8000" -ForegroundColor Green
}
Write-Host ""
Write-Host "Friend CV script should publish MQTT to this server IP on port 1883." -ForegroundColor Cyan
Write-Host "Staff devices only need the Network dashboard URL above." -ForegroundColor Cyan
Write-Host ""

Push-Location $Backend
& $BackendPython -m uvicorn main:app --host 0.0.0.0 --port 8000
Pop-Location
