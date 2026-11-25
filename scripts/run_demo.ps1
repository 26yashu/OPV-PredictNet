# ==============================================================================
# OPV-PredictNet — Local Demonstration Launcher (PowerShell)
# Starts FastAPI backend (port 8000) and Vite React frontend (port 5173).
# ==============================================================================

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "     OPV-PredictNet — Academic Demonstration Launcher    " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Project Root: $ProjectRoot`n"

# 1. Check Python Virtual Environment
$PythonExe = Join-Path $ProjectRoot "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "[WARNING] Python virtual environment not found at backend\.venv." -ForegroundColor Yellow
    Write-Host "Falling back to system python..."
    $PythonExe = "python"
}

# 2. Check Model Artifacts
$ModelsDir = Join-Path $ProjectRoot "ml\models\saved_models\phase9_v1"
$M3Model = Join-Path $ModelsDir "phase9_M3_model.joblib"
if (-not (Test-Path $M3Model)) {
    Write-Host "[INFO] Packaged model artifacts not detected. Running packaging script..." -ForegroundColor Yellow
    & $PythonExe (Join-Path $ProjectRoot "ml\scripts\package_models_phase9.py")
}

# 3. Check Node.js and Frontend dependencies
$FrontendDir = Join-Path $ProjectRoot "frontend"
$NodeModules = Join-Path $FrontendDir "node_modules"
if (-not (Test-Path $NodeModules)) {
    Write-Host "[INFO] Installing frontend dependencies..." -ForegroundColor Yellow
    Push-Location $FrontendDir
    npm install
    Pop-Location
}

Write-Host "`nStarting Backend FastAPI Service (http://127.0.0.1:8000)..." -ForegroundColor Green
$BackendProcess = Start-Process -FilePath $PythonExe `
    -ArgumentList "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload" `
    -WorkingDirectory $ProjectRoot `
    -PassThru

Write-Host "Starting Frontend Vite Development Server (http://localhost:5173)..." -ForegroundColor Green
$FrontendProcess = Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c", "npm", "run", "dev" `
    -WorkingDirectory $FrontendDir `
    -PassThru

Write-Host "`n----------------------------------------------------------" -ForegroundColor Cyan
Write-Host " Demonstration Services Running:" -ForegroundColor Cyan
Write-Host "   Frontend Application: http://localhost:5173" -ForegroundColor White
Write-Host "   Backend API Docs:     http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "   Backend Health Check: http://127.0.0.1:8000/api/health" -ForegroundColor White
Write-Host "----------------------------------------------------------" -ForegroundColor Cyan
Write-Host "Press Ctrl+C or close this window to terminate demonstration servers.`n"

try {
    # Keep script open until user interrupts
    Wait-Process -Id $BackendProcess.Id, $FrontendProcess.Id
} finally {
    Write-Host "`nStopping demonstration processes..." -ForegroundColor Yellow
    if ($BackendProcess -and -not $BackendProcess.HasExited) { Stop-Process -Id $BackendProcess.Id -Force -ErrorAction SilentlyContinue }
    if ($FrontendProcess -and -not $FrontendProcess.HasExited) { Stop-Process -Id $FrontendProcess.Id -Force -ErrorAction SilentlyContinue }
    Write-Host "Services stopped cleanly." -ForegroundColor Green
}
