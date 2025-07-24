@echo off
REM Batch Test runner script for the cart-pole PPO project
REM Provides an easy way to run different test suites

setlocal enabledelayedexpansion

if "%~1"=="" goto :show_help
if "%~1"=="help" goto :show_help
if "%~1"=="/?" goto :show_help

goto :main

:show_help
echo Cart-Pole PPO Test Runner (Windows Batch)
echo Usage: .\scripts\run_tests.bat [command]
echo.
echo Commands:
echo   all        - Run all tests
echo   fast       - Run only fast tests (exclude slow and integration)
echo   unit       - Run only unit tests
echo   integration - Run only integration tests
echo   coverage   - Run tests with coverage report
echo   verbose    - Run tests with verbose output
echo   [file]     - Run specific test file (e.g., test_agent.py)
echo.
echo Examples:
echo   .\scripts\run_tests.bat all
echo   .\scripts\run_tests.bat fast
echo   .\scripts\run_tests.bat test_environments.py
goto :eof

:check_dependencies
REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

REM Check if pytest is available
python -c "import pytest" >nul 2>&1
if errorlevel 1 (
    echo Error: pytest is not installed.
    echo Install test dependencies with: pip install -r requirements-test.txt
    exit /b 1
)
echo pytest is available
goto :eof

:run_tests
set cmd=%~1
set pytest_cmd=python -m pytest

if "%cmd%"=="all" (
    echo Running all tests...
    %pytest_cmd%
) else if "%cmd%"=="fast" (
    echo Running fast tests only...
    %pytest_cmd% -m "not slow and not integration"
) else if "%cmd%"=="unit" (
    echo Running unit tests only...
    %pytest_cmd% -m "unit"
) else if "%cmd%"=="integration" (
    echo Running integration tests only...
    %pytest_cmd% -m "integration"
) else if "%cmd%"=="coverage" (
    echo Running tests with coverage report...
    %pytest_cmd% --cov=src --cov-report=html --cov-report=term
) else if "%cmd%"=="verbose" (
    echo Running tests with verbose output...
    %pytest_cmd% -v
) else (
    set arg=%cmd%
    set prefix=!arg:~0,5!
    set suffix=!arg:~-3!
    setlocal enabledelayedexpansion
    if "!prefix!"=="test_" if "!suffix!"==".py" (
        echo Running specific test file: %cmd%
        %pytest_cmd% tests/%cmd%
        endlocal
        goto :eof
    )
    endlocal
    echo Unknown command: %cmd%
    goto :show_help
    exit /b 1
)
goto :eof

:main
REM Change to project root directory
cd /d "%~dp0.."

call :check_dependencies
if errorlevel 1 exit /b 1

echo Running tests from directory: %CD%
call :run_tests %~1
