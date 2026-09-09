#Requires -Version 5.1
<#
.SYNOPSIS
    Run the Regime Engine dashboard locally (FastAPI + Vite).
.DESCRIPTION
    - Checks Python 3.11+ and Node.js
    - Installs missing Python deps (requirements.txt)
    - Installs missing frontend deps (npm ci)
    - Starts FastAPI on :8000 and Vite dev server on :5173
    - Opens the dashboard in your browser
    - Ctrl+C stops both servers
#>

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvReady = $false

function Write-Step($Msg) { Write-Host "`n==> $Msg" -ForegroundColor Cyan }

# ---- 0. Prereq checks -------------------------------------------------------
Write-Step "Checking prerequisites"
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { throw "Python not found. Install Python 3.11+ and add it to PATH." }
$pyVer = & python --version 2>&1 | Out-String
Write-Host "  Python: $($pyVer.Trim())"

$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node) { throw "Node.js not found. Install Node 18+ from https://nodejs.org and add it to PATH." }
$nodeVer = & node --version 2>&1
Write-Host "  Node:   $($nodeVer.Trim())"

Set-Location $Root

# ---- 1. Python deps ---------------------------------------------------------
Write-Step "Checking Python dependencies"
# Portable check: try core imports; if missing, pip install
$imports = "fastapi uvicorn pandas numpy scipy sklearn torch pyarrow statsmodels yfinance"
$missing = @()
foreach ($mod in $imports.Split(' ')) {
    & python -c "import $mod" 2>$null
    if ($LASTEXITCODE -ne 0) { $missing += $mod }
}
if ($missing.Count -gt 0) {
    Write-Host "  Installing Python deps: $($missing -join ', ')...  (one-time, may take a few minutes)"
    & python -m pip install --upgrade pip --quiet
    & python -m pip install -r "requirements.txt"
    if ($LASTEXITCODE -ne 0) { throw "pip install failed. See output above." }
    Write-Host "  Python deps installed."
} else {
    Write-Host "  Python deps present."
}

# ---- 2. Frontend deps -------------------------------------------------------
Write-Step "Checking frontend dependencies"
if (-not (Test-Path "$Root\frontend\node_modules\.vite")) {
    Write-Host "  Installing frontend deps (npm ci)..."
    Push-Location "$Root\frontend"
    & npm ci --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { Pop-Location; throw "npm ci failed. See output above." }
    Pop-Location
    Write-Host "  Frontend deps installed."
} else {
    Write-Host "  Frontend deps present."
}

# ---- 3. Kill anything already on our ports ----------------------------------
foreach ($port in 8000, 5173) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conns) {
        Write-Host "  Port $port busy - stopping process(es) (PID $($conns.OwningProcess -join ', '))"
        $conns | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Milliseconds 800
    }
}

# ---- 4. Start backend -------------------------------------------------------
Write-Step "Starting FastAPI backend on http://localhost:8000"
$backend = Start-Process python -ArgumentList "api/run.py" -WorkingDirectory $Root `
    -PassThru -NoNewWindow -RedirectStandardOutput "$Root\api_server.log" -RedirectStandardError "$Root\api_server_err.log"
Write-Host "  backend PID $($backend.Id) (logs: api_server.log / api_server_err.log)"

# Wait for backend to accept connections
$backendUp = $false
for ($i = 0; $i -lt 40; $i++) {
    if ($backend.HasExited) { throw "Backend exited early. Check api_server_err.log" }
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8000/api/regime/current" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $backendUp = $true; break }
    } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $backendUp) { Write-Host "  (backend still warming up - continuing anyway)" }

# ---- 5. Start frontend ------------------------------------------------------
Write-Step "Starting Vite dev server on http://localhost:5173"
$frontend = Start-Process npm.cmd -ArgumentList "run", "dev" -WorkingDirectory "$Root\frontend" `
    -PassThru -NoNewWindow -RedirectStandardOutput "$Root\frontend\dev_server.log" -RedirectStandardError "$Root\frontend\dev_server_err.log"
Write-Host "  frontend PID $($frontend.Id) (logs: frontend/dev_server.log)"

# Wait for frontend
$frontendUp = $false
for ($i = 0; $i -lt 40; $i++) {
    if ($frontend.HasExited) { break }
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:5173" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $frontendUp = $true; break }
    } catch { Start-Sleep -Milliseconds 500 }
}

# ---- 6. Open browser --------------------------------------------------------
if ($frontendUp) {
    Write-Step "Dashboard ready"
    Write-Host ""
    Write-Host "  Dashboard : http://localhost:5173"
    Write-Host "  API docs  : http://localhost:8000/docs"
    Write-Host ""
    Start-Process "http://localhost:5173"
    Write-Host "  Opened in your browser."
    Write-Host "  Press Ctrl+C to stop both servers.`n"
} else {
    Write-Host "  Frontend still starting - open http://localhost:5173 manually."
}

# ---- 7. Keep alive until Ctrl+C --------------------------------------------
try {
    while ($true) { Start-Sleep -Seconds 1 }
} finally {
    Write-Host "`nStopping servers..."
    foreach ($p in @($backend, $frontend)) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
    Write-Host "Done."
}