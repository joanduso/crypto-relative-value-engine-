@echo off
setlocal
cd /d "%~dp0"

start "" http://localhost:8080
python mobile_dashboard.py
