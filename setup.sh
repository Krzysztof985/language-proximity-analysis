#!/bin/bash
# Setup script for language-proximity-analysis project

set -e  # Exit on error

echo "=========================================="
echo "Setting up language-proximity-analysis"
echo "=========================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Display Python version
PYTHON_VERSION=$(python3 --version)
echo "Using $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data
mkdir -p models
mkdir -p logs
mkdir -p results

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
