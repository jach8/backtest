#!/usr/bin/env python3
"""
Test runner script for the backtest package.

Usage:
    python run_tests.py [options]

Options:
    --unit          Run only unit tests
    --integration   Run only integration tests
    --coverage      Generate coverage report
    --html         Generate HTML coverage report
    --all          Run all tests with coverage (default)
"""

import sys
import subprocess
import argparse

def run_tests(args):
    """Run tests based on provided arguments."""
    base_command = ["pytest"]
    
    # Add coverage options if requested
    if args.coverage or args.html:
        base_command.extend([
            "--cov=backtest",
            "--cov-report=term-missing",
        ])
        if args.html:
            base_command.append("--cov-report=html")
    
    # Add test selection based on type
    if args.unit:
        base_command.extend(["-m", "unit"])
    elif args.integration:
        base_command.extend(["-m", "integration"])
    
    # Add verbosity
    base_command.append("-v")
    
    try:
        subprocess.run(base_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code: {e.returncode}")
        sys.exit(e.returncode)

def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run backtest package tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--all", action="store_true", help="Run all tests with coverage (default)")
    
    args = parser.parse_args()
    
    # If no specific test type is selected, run all tests
    if not any([args.unit, args.integration, args.coverage, args.html]):
        args.all = True
    
    # If --all is specified, enable coverage
    if args.all:
        args.coverage = True
    
    run_tests(args)

if __name__ == "__main__":
    main()