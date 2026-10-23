@echo off
setlocal
title BNS Legal Intelligence Launcher
cd /d "%~dp0"

echo ========================================================
echo   BNS Legal Intelligence - Multi-Agent RAG
echo   http://127.0.0.1:8000
echo ========================================================

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating Python virtual environment...
  python -m venv .venv
  if errorlevel 1 goto :python_error
)

call ".venv\Scripts\activate.bat"
echo [INFO] Installing/checking Python dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 goto :python_error

if not exist "frontend\dist\index.html" (
  where npm >nul 2>nul
  if errorlevel 1 (
    echo [WARN] Node.js/npm not found. The API will start, but the UI must be built separately.
  ) else (
    echo [INFO] Building React frontend...
    pushd frontend
    call npm install
    if errorlevel 1 (popd & goto :frontend_error)
    call npm run build
    if errorlevel 1 (popd & goto :frontend_error)
    popd
  )
)

python app.py
goto :end

:python_error
echo [ERROR] Python setup failed. Check that Python is installed and on PATH.
pause
goto :end

:frontend_error
echo [ERROR] Frontend setup/build failed. Install a supported Node.js version and retry.
pause
goto :end

:end
pause
endlocal
