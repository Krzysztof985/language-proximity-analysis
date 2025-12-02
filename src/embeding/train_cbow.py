#!/usr/bin/env python3
"""
Training script for CBOW Word2Vec model.
"""
import argparse
import os
import sys
import json
import torch

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.logging.logging_config import setup_logger
logger = setup_logger(__name__, "train_cbow.log")
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.embeding.cbow import CBOWModel, CBOWDataset, train_cbow

from src.embeding import hyperparamiters as hp

def train_model():
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, hp.DATA_DIR))
    output_dir = os.path.normpath(os.path.join(script_dir, hp.OUTPUT_DIR))
    
    # Set device
    if hp.DEVICE == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = hp.DEVICE
    
    logger.info("=" * 60)
    logger.info("CBOW Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Languages: {hp.LANGUAGES}")
    logger.info(f"Data type: {hp.DATA_TYPE}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Embedding dimension: {hp.EMBEDDING_DIM}")
    logger.info(f"Epochs: {hp.EPOCHS}")
    logger.info(f"Batch size: {hp.BATCH_SIZE}")
    logger.info(f"Learning rate: {hp.LEARNING_RATE}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)
    
    # Load data
    logger.info("\nLoading data...")
    data_path = os.path.join(os.path.dirname(script_dir), "data")
    sys.path.insert(0, data_path)
    
    if hp.DATA_TYPE == 'words':
        from datasets.multilingual_dataset import MultilingualWordDataset
        dataset_class = MultilingualWordDataset
    else:
        from datasets.multilingual_dataset import MultilingualPhonemeDataset
        dataset_class = MultilingualPhonemeDataset
    
    multilang_dataset = dataset_class(hp.LANGUAGES, data_dir)
    logger.info(f"Loaded {len(multilang_dataset)} tokens")
    
    # Extract sequences (words or phoneme lists)
    sequences = [seq for seq, _ in multilang_dataset.samples]
    if hp.MAX_TOKENS:
        # Note: MAX_TOKENS now limits number of words/sequences, not individual characters
        sequences = sequences[:hp.MAX_TOKENS]
        logger.info(f"Using first {len(sequences)} sequences")
    
    # Calculate maximum word length to use as window size
    max_word_length = max(len(seq) for seq in sequences)
    window_size = max_word_length
    logger.info(f"\nCalculated window size from max word length: {window_size}")
    
    # Create CBOW dataset
    logger.info(f"\nCreating CBOW dataset with window size {window_size}...")
    cbow_dataset = CBOWDataset(sequences, window_size=window_size)
    logger.info(f"Vocabulary size: {cbow_dataset.vocab_size}")
    logger.info(f"Training samples: {len(cbow_dataset)}")
    
    # Create model
    logger.info(f"\nCreating model with embedding dimension {hp.EMBEDDING_DIM}...")
    model = CBOWModel(cbow_dataset.vocab_size, hp.EMBEDDING_DIM)
    
    # Train
    logger.info(f"\nTraining on {device}...")
    model, losses = train_cbow(
        model,
        cbow_dataset,
        epochs=hp.EPOCHS,
        batch_size=hp.BATCH_SIZE,
        learning_rate=hp.LEARNING_RATE,
        device=device
    )
    
    # Save model and training info
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"cbow_{hp.DATA_TYPE}_{'_'.join(hp.LANGUAGES)}_{timestamp}"
    
    # Save model
    model_path = os.path.join(output_dir, f"{model_name}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': cbow_dataset.vocab_size,
        'embedding_dim': hp.EMBEDDING_DIM,
        'word_to_idx': cbow_dataset.word_to_idx,
        'idx_to_word': cbow_dataset.idx_to_word,
    }, model_path)
    logger.info(f"\nModel saved to: {model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump({
            'config': {
                'languages': hp.LANGUAGES,
                'data_type': hp.DATA_TYPE,
                'embedding_dim': hp.EMBEDDING_DIM,
                'window_size': window_size,
                'epochs': hp.EPOCHS,
                'batch_size': hp.BATCH_SIZE,
                'learning_rate': hp.LEARNING_RATE,
            },
            'losses': losses,
            'vocab_size': cbow_dataset.vocab_size,
            'training_samples': len(cbow_dataset),
        }, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    train_model()
