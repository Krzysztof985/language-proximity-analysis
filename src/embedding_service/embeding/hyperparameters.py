import json
import os

# Load config
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
config_path = os.path.join(project_root, 'config.json')

with open(config_path, 'r') as f:
    config = json.load(f)

es_config = config['embedding_service']

# Data parameters
LANGUAGES = es_config['languages']
DATA_TYPE = 'phonemes'  # 'words' or 'phonemes'
DATA_DIR = es_config['directories']['data']
MAX_TOKENS = None

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1 # they must add to 1

# Model parameters
EMBEDDING_DIM = 500
# WINDOW_SIZE = 2 # now calculated from max word length

# Training parameters
EPOCHS = 30
BATCH_SIZE = 65536
LEARNING_RATE = 0.001
PATIENCE = 3
MIN_DELTA = 0.001
DEVICE = 'auto' 
"""'auto', 'cpu', 'cuda' i sugget to use 'auto' 
    for now piplane suports only nvidia cards or cpu's"""

# Output parameters
OUTPUT_DIR = es_config['directories']['models']
SAVE_FREQ = 0  # 0 = only final
