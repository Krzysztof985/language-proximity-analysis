#!/bin/bash
set -e

SESSION="training_lpa_session"

# Kill existing session
if screen -list | grep -q "\.${SESSION}"; then
    screen -S "$SESSION" -X quit
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi  


# Activate venv early
source .venv/bin/activate

# Resolve languages
if [ -z "$@" ]; then
    LANGUAGES=$(python -c "import json; print(' '.join(json.load(open('config.json'))['embedding_service']['languages']))")
else
    LANGUAGES="$@"
fi

# Run pipeline as MODULE
screen -dmS "$SESSION" bash -c "
set -e
source .venv/bin/activate
python src/embedding_service/run_train_data_pipeline.py $LANGUAGES
"

echo "Training started in screen session: $SESSION"
echo "Attach with: screen -r $SESSION"
