@echo off
title VisionForge AI
if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found.
    echo Please run setup_windows.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate
streamlit run app.py
pause
