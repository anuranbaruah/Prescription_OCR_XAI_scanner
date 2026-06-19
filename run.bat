@echo off
REM ===================================================================
REM  Explainable Prescription Analyzer - one-click launcher
REM  Starts the FastAPI backend (:8000) and the React/Vite frontend
REM  (:5173) in separate windows, then opens the app in your browser.
REM  Close either window (or press a key here) to stop.
REM ===================================================================
setlocal
cd /d "%~dp0"
title Prescription Analyzer - launcher

set "PYEXE=%~dp0backend\.venv\Scripts\python.exe"

echo.
echo  Explainable Prescription Analyzer
echo  =================================
echo.

REM --- sanity checks -------------------------------------------------
if not exist "%PYEXE%" (
  echo [ERROR] Backend virtualenv not found at:
  echo         %PYEXE%
  echo         Create it first, e.g.:  cd backend ^&^& python -m venv .venv
  echo.
  pause
  exit /b 1
)

if not exist "%~dp0frontend\node_modules" (
  echo [setup] frontend\node_modules missing - running "npm install" once...
  pushd "%~dp0frontend"
  call npm install
  popd
  echo.
)

REM --- launch backend ------------------------------------------------
echo [1/3] Starting backend  -> http://localhost:8000
start "Rx Analyzer - Backend" cmd /k "cd /d "%~dp0backend" && ".venv\Scripts\python.exe" -m uvicorn app.main:app --port 8000"

REM --- launch frontend -----------------------------------------------
echo [2/3] Starting frontend -> http://localhost:5173
start "Rx Analyzer - Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

REM --- open the browser once Vite has had a moment to boot -----------
echo [3/3] Opening browser in a few seconds...
timeout /t 6 /nobreak >nul
start "" "http://localhost:5173"

echo.
echo  Both servers are starting in their own windows.
echo  - App / UI : http://localhost:5173
echo  - API docs : http://localhost:8000/docs
echo.
echo  To stop: close the Backend and Frontend windows.
echo.
pause
endlocal
