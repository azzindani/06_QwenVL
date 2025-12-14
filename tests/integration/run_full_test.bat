@echo off
REM Comprehensive Test Runner - Tests ALL handlers
REM Usage: run_full_test.bat [--quick] [--verbose] [--handler name]

echo ============================================================
echo   Qwen VL Comprehensive Handler Test Suite
echo ============================================================
echo.

REM Activate conda environment
call conda activate omni_env

REM Change to project directory
cd /d "%~dp0\..\.."

REM Run comprehensive tests
python tests/integration/test_all_handlers.py %*

echo.
echo ============================================================
echo   Test Complete - Check tests/results/ for JSON report
echo ============================================================
pause
