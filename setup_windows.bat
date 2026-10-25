@echo off
title VisionForge AI - Setup
echo ==============================================
echo   VisionForge AI - YOLOv8 Project Setup
echo ==============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found in PATH.
    echo Please install Python 3.10 or 3.11 and enable "Add Python to PATH".
    pause
    exit /b 1
)

echo Creating virtual environment...
python -m venv .venv

echo Activating environment...
call .venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing project dependencies...
pip install -r requirements.txt

echo.
echo ==============================================
echo Setup completed successfully.
echo Run the project using run_app.bat
echo ==============================================
pause
