@echo off
setlocal
cd /d "%~dp0"

start "" http://localhost:8501
powershell -ExecutionPolicy Bypass -File ".\start_services.ps1"
