#!/bin/bash

# Script to run all CI checks locally before committing
# This ensures your changes will pass the automated checks in the CI pipeline

set -e  # Exit on first error

echo "=========================================="
echo "Running CI Checks"
echo "=========================================="
echo ""

# Change to project root directory
cd "$(git rev-parse --show-toplevel)"

# Track overall status
FAILED=0

pip install --upgrade pip
pip install --upgrade -e ".[dev]"

# Style Checks
echo "▶ Running pydocstyle..."
if pydocstyle src; then
    echo "✓ pydocstyle passed"
else
    echo "✗ pydocstyle failed"
    FAILED=1
fi
echo ""

echo "▶ Running black..."
if black .; then
    echo "✓ black passed"
else
    echo "✗ black failed"
    FAILED=1
fi
echo ""

# Static Type Analysis
echo "▶ Running pyright..."
if pyright src; then
    echo "✓ pyright passed"
else
    echo "✗ pyright failed"
    FAILED=1
fi
echo ""

# Unit Tests and Coverage
echo "▶ Running pytest with coverage..."
if pytest -q --cov=src --cov-report=term-missing --cov-fail-under=70; then
    echo "✓ pytest passed"
else
    echo "✗ pytest failed"
    FAILED=1
fi
echo ""

# Summary
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo "✓ All CI checks passed!"
    echo "=========================================="
    exit 0
else
    echo "✗ Some CI checks failed"
    echo "=========================================="
    exit 1
fi
