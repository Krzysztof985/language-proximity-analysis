# Data parameters
LANGUAGES = ['pl']
DATA_TYPE = 'words'  # 'words' or 'phonemes'
DATA_DIR = '../../data'
MAX_TOKENS = None

# Model parameters
EMBEDDING_DIM = 100
WINDOW_SIZE = 2

# Training parameters
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
DEVICE = 'auto'  # 'auto', 'cpu', 'cuda'

# Output parameters
OUTPUT_DIR = '../../models'
SAVE_FREQ = 0  # 0 = only final
