#!/bin/bash
# Build script for language-proximity-analysis project
# This script trains both phoneme and word models

set -e  # Exit on error

echo "=========================================="
echo "Building language-proximity-analysis"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Default languages
LANGUAGES=${@:-pl en}

echo "Training models for languages: $LANGUAGES"
echo ""

# Run training pipeline
echo "Running training pipeline..."
python src/embedding_service/run_train_data_pipeline.py $LANGUAGES

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo ""
echo "Trained models are saved in the 'models/' directory."
echo ""
