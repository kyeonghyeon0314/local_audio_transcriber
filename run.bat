@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title Local Multimodal Transcriber

if not exist ".venv\Scripts\python.exe" (
    echo 먼저 setup.bat을 실행해 설치를 완료해 주세요.
    pause
    exit /b 1
)

set "PYTHONUTF8=1"
".venv\Scripts\python.exe" run.py %*

echo.
pause
