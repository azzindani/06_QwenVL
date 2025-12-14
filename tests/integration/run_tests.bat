@echo off
REM Batch script to run real device integration tests
REM Usage: run_tests.bat [task] [--quick]

echo ============================================================
echo   Qwen VL Real Device Integration Tests
echo ============================================================
echo.

REM Activate conda environment
call conda activate omni_env

REM Change to project directory
cd /d "%~dp0\..\.."

REM Run tests
if "%1"=="" (
    echo Running all tests...
    python tests/integration/test_real_inference.py %2 %3
) else if "%1"=="--quick" (
    echo Running quick tests...
    python tests/integration/test_real_inference.py --quick
) else (
    echo Running %1 tests...
    python tests/integration/test_real_inference.py --task %1 %2 %3
)

echo.
echo ============================================================
echo   Tests Complete
echo ============================================================
pause
