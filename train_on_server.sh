#!/bin/bash
# Build script for language-proximity-analysis project
# This script trains phoneme and word models in a safe way that keeps running
# even after closing VS Code or ending SSH sessions.

set -e  # Exit on error

mkdir -p logs

echo "=========================================="
echo "Starting language-proximity-analysis build"
echo "This process will continue even if you close VS Code."
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

# Default languages from config.json
if [ -z "$@" ]; then
    LANGUAGES=$(python -c "import json; print(' '.join(json.load(open('config.json'))['embedding_service']['languages']))")
else
    LANGUAGES=$@
fi

echo "Training models for languages: $LANGUAGES"
echo ""

# Prepare logs directory
mkdir -p logs
LOGFILE="logs/train_$(date +%Y-%m-%d_%H-%M-%S).log"

echo "Running training pipeline in background..."
echo "Logs will be saved to: $LOGFILE"
echo ""

# Run the pipeline safely in background (immune to VS Code closing)
nohup python src/embedding_service/run_train_data_pipeline.py $LANGUAGES \
    > "$LOGFILE" 2>&1 &

PID=$!

STATUS_LOG="logs/build_status.log"

{
echo "=========================================="
echo "Training started!"
echo "Background process PID: $PID"
echo "You can close VS Code safely — training will continue."
echo "To monitor progress:"
echo "  tail -f $LOGFILE"
echo ""
echo "To stop the training:"
echo "  kill $PID"
echo "=========================================="
} | tee "$STATUS_LOG" > /dev/null