#!/bin/bash
# Test script for language-proximity-analysis project

set -e  # Exit on error

echo "=========================================="
echo "Running tests for language-proximity-analysis"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Set PYTHONPATH to include src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "PYTHONPATH: $PYTHONPATH"
echo ""

# Run dataset verification tests
if [ -f "tests/verify_datasets.py" ]; then
    echo "Running dataset verification tests..."
    python tests/verify_datasets.py
    echo ""
fi

# Run CBOW tests
if [ -f "tests/test_cbow.py" ]; then
    echo "Running CBOW model tests..."
    python tests/test_cbow.py
    echo ""
fi

# Run Model Comparison tests
if [ -f "tests/test_models.py" ]; then
    echo "Running Model Comparison tests..."
    python tests/test_models.py
    echo ""
fi

# If pytest is installed, run it
if command -v pytest &> /dev/null; then
    echo "Running pytest..."
    pytest tests/ -v
    echo ""
else
    echo "Note: pytest is not installed. Install it with 'pip install pytest' for more comprehensive testing."
fi

echo "=========================================="
echo "All tests completed!"
echo "=========================================="
