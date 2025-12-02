# Data parameters
LANGUAGES = ['pl']
DATA_TYPE = 'phonemes'  # 'words' or 'phonemes'
DATA_DIR = '../../data'
MAX_TOKENS = None

# Model parameters
EMBEDDING_DIM = 100
# WINDOW_SIZE = 2 # now calculated from max word length

# Training parameters
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
DEVICE = 'auto'  # 'auto', 'cpu', 'cuda' i sugget to use 'auto'

# Output parameters
OUTPUT_DIR = '../../models'
SAVE_FREQ = 0  # 0 = only final
