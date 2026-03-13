$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

powershell -ExecutionPolicy Bypass -File (Join-Path $ProjectRoot "start_services.ps1")
