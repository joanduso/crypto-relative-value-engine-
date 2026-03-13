$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = "C:\Users\Jose.Duran\AppData\Local\Programs\Python\Python312\python.exe"
$OutputDir = Join-Path $ProjectRoot "output"
$DashboardPort = 8501
$StreamlitPidFile = Join-Path $OutputDir "streamlit.pid"
$MonitorPidFile = Join-Path $OutputDir "monitor.pid"

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

function Test-PidFileActive {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PidFile
    )

    if (-not (Test-Path $PidFile)) {
        return $false
    }

    $pidValue = Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $pidValue) {
        return $false
    }

    $running = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
    return $null -ne $running
}

function Test-PortListening {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Port
    )

    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    return $null -ne $listener
}

if (-not (Test-Path $PythonExe)) {
    throw "Python 3.12 no encontrado en $PythonExe"
}

$streamlitActive = (Test-PortListening -Port $DashboardPort) -or (Test-PidFileActive -PidFile $StreamlitPidFile)
if (-not $streamlitActive) {
    $streamlitProcess = Start-Process -FilePath $PythonExe `
        -ArgumentList @(
            "-m", "streamlit", "run", "local_dashboard.py",
            "--server.headless", "true",
            "--server.port", "$DashboardPort"
        ) `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput (Join-Path $OutputDir "streamlit_stdout.log") `
        -RedirectStandardError (Join-Path $OutputDir "streamlit_stderr.log") `
        -WindowStyle Hidden `
        -PassThru

    Set-Content -Path $StreamlitPidFile -Value $streamlitProcess.Id

    Start-Sleep -Seconds 4
}

if (-not (Test-PidFileActive -PidFile $MonitorPidFile)) {
    $monitorProcess = Start-Process -FilePath $PythonExe `
        -ArgumentList @("monitor.py", "--mode", "COPILOT", "--poll-minutes", "5") `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput (Join-Path $OutputDir "monitor_stdout.log") `
        -RedirectStandardError (Join-Path $OutputDir "monitor_stderr.log") `
        -WindowStyle Hidden `
        -PassThru

    Set-Content -Path $MonitorPidFile -Value $monitorProcess.Id
}

if (Test-PortListening -Port $DashboardPort) {
    Start-Process "http://localhost:$DashboardPort"
}
