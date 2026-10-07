@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo 먼저 setup.bat을 실행해 주세요.
    pause
    exit /b 1
)
set "PYTHONUTF8=1"
".venv\Scripts\python.exe" tools\diagnose_audio.py
echo.
pause
