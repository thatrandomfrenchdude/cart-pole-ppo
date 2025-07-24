# PowerShell Test runner script for the cart-pole PPO project
# Provides an easy way to run different test suites

param(
    [Parameter(Position=0)]
    [string]$Command = ""
)

function Show-Help {
    Write-Host "Cart-Pole PPO Test Runner" -ForegroundColor Green
    Write-Host "Usage: .\scripts\run_tests.ps1 [command]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Cyan
    Write-Host "  all        - Run all tests"
    Write-Host "  fast       - Run only fast tests (exclude slow and integration)"
    Write-Host "  unit       - Run only unit tests"
    Write-Host "  integration - Run only integration tests"
    Write-Host "  coverage   - Run tests with coverage report"
    Write-Host "  verbose    - Run tests with verbose output"
    Write-Host "  [file]     - Run specific test file (e.g., test_agent.py)"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  .\scripts\run_tests.ps1 all"
    Write-Host "  .\scripts\run_tests.ps1 fast"
    Write-Host "  .\scripts\run_tests.ps1 test_environments.py"
}

function Test-Dependencies {
    # Check if Python is available
    try {
        $pythonVersion = python --version 2>$null
        if (-not $pythonVersion) {
            Write-Error "Error: Python is not installed or not in PATH"
            exit 1
        }
        Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
    }
    catch {
        Write-Error "Error: Python is not installed or not in PATH"
        exit 1
    }
    
    # Check if pytest is available
    try {
        $pytestResult = python -c "import pytest; print('pytest available')" 2>$null
        if (-not $pytestResult) {
            Write-Error "Error: pytest is not installed."
            Write-Host "Install test dependencies with: pip install -r requirements-test.txt" -ForegroundColor Yellow
            exit 1
        }
        Write-Host "pytest is available" -ForegroundColor Green
    }
    catch {
        Write-Error "Error: pytest is not installed."
        Write-Host "Install test dependencies with: pip install -r requirements-test.txt" -ForegroundColor Yellow
        exit 1
    }
}

function Invoke-Tests {
    param([string]$TestCommand)
    
    # Base pytest command
    $pytestCmd = "python -m pytest"
    
    switch ($TestCommand) {
        "all" {
            Write-Host "Running all tests..." -ForegroundColor Green
            Invoke-Expression $pytestCmd
        }
        "fast" {
            Write-Host "Running fast tests only..." -ForegroundColor Green
            Invoke-Expression "$pytestCmd -m `"not slow and not integration`""
        }
        "unit" {
            Write-Host "Running unit tests only..." -ForegroundColor Green
            Invoke-Expression "$pytestCmd -m `"unit`""
        }
        "integration" {
            Write-Host "Running integration tests only..." -ForegroundColor Green
            Invoke-Expression "$pytestCmd -m `"integration`""
        }
        "coverage" {
            Write-Host "Running tests with coverage report..." -ForegroundColor Green
            Invoke-Expression "$pytestCmd --cov=src --cov-report=html --cov-report=term"
        }
        "verbose" {
            Write-Host "Running tests with verbose output..." -ForegroundColor Green
            Invoke-Expression "$pytestCmd -v"
        }
        default {
            if ($TestCommand -match "^test_.*\.py$") {
                Write-Host "Running specific test file: $TestCommand" -ForegroundColor Green
                Invoke-Expression "$pytestCmd tests/$TestCommand"
            }
            else {
                Write-Error "Unknown command: $TestCommand"
                Show-Help
                exit 1
            }
        }
    }
}

function Main {
    # Change to project root directory
    $scriptPath = $PSCommandPath
    if (-not $scriptPath) {
        $scriptPath = $MyInvocation.MyCommand.Path
    }
    if (-not $scriptPath) {
        # Fallback to current directory
        $scriptDir = Get-Location
        $projectRoot = Split-Path -Parent $scriptDir
    } else {
        $scriptDir = Split-Path -Parent $scriptPath
        $projectRoot = Split-Path -Parent $scriptDir
    }
    Set-Location $projectRoot
    
    if (-not $Command -or $Command -eq "") {
        Show-Help
        return
    }
    
    Test-Dependencies
    
    Write-Host "Running tests from directory: $(Get-Location)" -ForegroundColor Cyan
    Invoke-Tests $Command
}

# Execute main function
Main
