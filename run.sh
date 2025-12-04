#!/bin/bash
# Run script for language-proximity-analysis project
# This script runs the word comparison tool

set -e  # Exit on error

echo "=========================================="
echo "Running language-proximity-analysis"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Set PYTHONPATH to include project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models/*.pt 2>/dev/null)" ]; then
    echo "Warning: No trained models found in 'models/' directory."
    echo "Please run ./build.sh first to train the models."
    exit 1
fi

# Run word comparison
echo "Running word comparison tool..."
echo ""

# You can modify the default values below or pass them as arguments
python src.main

echo ""
echo "=========================================="
echo "Execution completed!"
echo "=========================================="
