
# Data parameters
LANGUAGES = ['pl', 'en', 'de']
DATA_TYPE = 'phonemes'  # 'words' or 'phonemes'
DATA_DIR = '../../data'
MAX_TOKENS = None

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1 # they must add to 1

# Model parameters
EMBEDDING_DIM = 100
# WINDOW_SIZE = 2 # now calculated from max word length

# Training parameters
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
PATIENCE = 3
MIN_DELTA = 0.001
DEVICE = 'auto' 
"""'auto', 'cpu', 'cuda' i sugget to use 'auto' 
    for now piplane suports only nvidia cards or cpu's"""

# Output parameters
OUTPUT_DIR = '../../models'
SAVE_FREQ = 0  # 0 = only final
