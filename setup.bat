@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title Local Multimodal Transcriber - 설치

echo ============================================================
echo  Local Multimodal Transcriber 설치
echo ============================================================
echo.

set "PYEXE="
py -3 --version >nul 2>&1 && set "PYEXE=py -3"
if not defined PYEXE (
    python --version >nul 2>&1 && set "PYEXE=python"
)
if not defined PYEXE (
    echo [오류] Python을 찾을 수 없습니다.
    echo https://www.python.org/downloads/ 에서 Python 3.12를 설치하고
    echo 설치 화면에서 "Add python.exe to PATH"를 체크한 뒤 다시 실행해 주세요.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] 가상환경 생성 중...
    %PYEXE% -m venv .venv || goto :fail
)
set "VPY=.venv\Scripts\python.exe"

echo [2/4] pip 업데이트...
"%VPY%" -m pip install --upgrade pip || goto :fail

echo [3/4] PyTorch (CUDA 12.8, RTX 50 시리즈 지원) 설치 중... 수 GB라 시간이 걸립니다.
"%VPY%" -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :fail

echo [4/4] 나머지 패키지 설치 중...
"%VPY%" -m pip install --upgrade -r requirements.txt || goto :fail

echo.
"%VPY%" -c "import torch; print('CUDA 사용 가능:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU 없음')"

echo.
echo ============================================================
echo  Hugging Face 로그인 (최초 1회)
echo ============================================================
echo  Gemma 모델은 사용 동의가 필요합니다.
echo   1) https://huggingface.co/google/gemma-4-E4B-it 에서 라이선스 동의
echo   2) https://huggingface.co/settings/tokens 에서 Read 토큰 생성
echo   3) 아래에 토큰 붙여넣기 (이미 로그인했다면 이 창을 닫아도 됩니다)
echo.
".venv\Scripts\hf.exe" auth login

echo.
echo 설치 완료! input 폴더에 녹음과 PDF를 넣고 run.bat을 실행하세요.
pause
exit /b 0

:fail
echo.
echo [오류] 설치 중 문제가 발생했습니다. 위 메시지를 확인해 주세요.
pause
exit /b 1
