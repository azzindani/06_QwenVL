@echo off
REM Interactive Demo Script
REM Usage: interactive.bat

echo ============================================================
echo   Qwen VL Interactive Demo
echo ============================================================
echo.

REM Activate conda environment
call conda activate omni_env

REM Change to project directory
cd /d "%~dp0\..\.."

REM Run interactive demo
python tests/integration/interactive_demo.py

pause
