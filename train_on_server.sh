#!/bin/bash
# Build script for language-proximity-analysis project
# Runs the training in a persistent tmux session.

set -e

echo "=========================================="
echo "Starting language-proximity-analysis build"
echo "This process will continue even if you close VS Code."
echo "=========================================="

# Ensure logs directory exists
mkdir -p logs

# Name of tmux session
SESSION="training"

# Kill previous session if exists (optional)
if tmux has-session -t $SESSION 2>/dev/null; then
    echo "Existing tmux session '$SESSION' found. Killing it..."
    tmux kill-session -t $SESSION
fi

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate environment
source .venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Determine languages
if [ -z "$@" ]; then
    LANGUAGES=$(python -c "import json; print(' '.join(json.load(open('config.json'))['embedding_service']['languages']))")
else
    LANGUAGES=$@
fi

echo "Training models for languages: $LANGUAGES"
echo ""

# Logfile for training
LOGFILE="logs/train_$(date +%Y-%m-%d_%H-%M-%S).log"

echo "Starting tmux session '$SESSION'..."
tmux new-session -d -s $SESSION

echo "Running training pipeline inside tmux..."

tmux send-keys -t $SESSION "
python src/embedding_service/run_train_data_pipeline.py $LANGUAGES | tee $LOGFILE
" C-m

# Write status log (overwrite)
STATUS_LOG="logs/build_status.log"
{
echo "=========================================="
echo "Training started inside tmux session: $SESSION"
echo "Log file: $LOGFILE"
echo ""
echo "To reattach to training session:"
echo "  tmux attach -t $SESSION"
echo ""
echo "To detach: Ctrl+B, then D"
echo ""
echo "To stop training:"
echo "  tmux kill-session -t $SESSION"
echo "=========================================="
} | tee "$STATUS_LOG" > /dev/null

echo ""
echo "=========================================="
echo "Build initialized successfully!"
echo "Training is running in tmux session: $SESSION"
echo "You can safely close VS Code now."
echo "=========================================="
