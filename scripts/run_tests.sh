#!/bin/bash
# Test runner script for the cart-pole PPO project
# Provides an easy way to run different test suites

set -e

show_help() {
    echo "Cart-Pole PPO Test Runner"
    echo "Usage: ./scripts/run_tests.sh [command]"
    echo ""
    echo "Commands:"
    echo "  all       - Run all tests"
    echo "  fast      - Run only fast tests (exclude slow and integration)"
    echo "  unit      - Run only unit tests"
    echo "  integration - Run only integration tests"
    echo "  coverage  - Run tests with coverage report"
    echo "  verbose   - Run tests with verbose output"
    echo "  [file]    - Run specific test file (e.g., test_agent.py)"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run_tests.sh all"
    echo "  ./scripts/run_tests.sh fast"
    echo "  ./scripts/run_tests.sh test_environments.py"
}

check_dependencies() {
    if ! command -v python3 &> /dev/null; then
        echo "Error: Python3 is not installed or not in PATH"
        exit 1
    fi
    
    # Check if pytest is available
    if ! python3 -c "import pytest" 2>/dev/null; then
        echo "Error: pytest is not installed."
        echo "Install test dependencies with: pip install -r requirements-test.txt"
        exit 1
    fi
}

run_tests() {
    local cmd="$1"
    
    # Base pytest command
    local pytest_cmd="python3 -m pytest"
    
    case "$cmd" in
        "all")
            echo "Running all tests..."
            $pytest_cmd
            ;;
        "fast")
            echo "Running fast tests only..."
            $pytest_cmd -m "not slow and not integration"
            ;;
        "unit")
            echo "Running unit tests only..."
            $pytest_cmd -m "unit"
            ;;
        "integration")
            echo "Running integration tests only..."
            $pytest_cmd -m "integration"
            ;;
        "coverage")
            echo "Running tests with coverage report..."
            $pytest_cmd --cov=src --cov-report=html --cov-report=term
            ;;
        "verbose")
            echo "Running tests with verbose output..."
            $pytest_cmd -v
            ;;
        test_*.py)
            echo "Running specific test file: $cmd"
            $pytest_cmd "tests/$cmd"
            ;;
        *)
            echo "Unknown command: $cmd"
            show_help
            exit 1
            ;;
    esac
}

main() {
    # Change to project root directory
    cd "$(dirname "$0")/.."
    
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    check_dependencies
    
    echo "Running tests from directory: $(pwd)"
    run_tests "$1"
}

main "$@"
