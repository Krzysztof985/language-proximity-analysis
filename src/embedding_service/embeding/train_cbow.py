#!/usr/bin/env python3
"""
Training script for CBOW Word2Vec model.
"""
import argparse
import os
import sys
import json
import torch
import random
import numpy as np

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.logger.logging_config import setup_logger
logger = setup_logger(__name__, "train_cbow.log")
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.embedding_service.embeding.cbow import CBOWModel, CBOWDataset, train_cbow, build_vocab, evaluate_model

from src.embedding_service.embeding import hyperparamiters as hp

def train_model(languages=None, data_type=None):
    # Use provided arguments or fallback to hyperparameters
    languages = languages if languages is not None else hp.LANGUAGES
    data_type = data_type if data_type is not None else hp.DATA_TYPE
    
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
    logger.info(f"Languages: {languages}")
    logger.info(f"Data type: {data_type}")
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
    
    if data_type == 'words':
        from datasets.multilingual_dataset import MultilingualWordDataset
        dataset_class = MultilingualWordDataset
    else:
        from datasets.multilingual_dataset import MultilingualPhonemeDataset
        dataset_class = MultilingualPhonemeDataset
    
    multilang_dataset = dataset_class(languages, data_dir)
    logger.info(f"Loaded {len(multilang_dataset)} tokens")
    
    # Extract sequences (words or phoneme lists)
    sequences = [seq for seq, _ in multilang_dataset.samples]
    if hp.MAX_TOKENS:
        # Note: MAX_TOKENS now limits number of words/sequences, not individual characters
        sequences = sequences[:hp.MAX_TOKENS]
        logger.info(f"Using first {len(sequences)} sequences")
    
    # Shuffle sequences
    random.seed(42)
    random.shuffle(sequences)
    
    # Split data
    total_sequences = len(sequences)
    train_size = int(total_sequences * hp.TRAIN_RATIO)
    val_size = int(total_sequences * hp.VAL_RATIO)
    test_size = total_sequences - train_size - val_size
    
    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:train_size + val_size]
    test_sequences = sequences[train_size + val_size:]
    
    logger.info(f"\nData Split:")
    logger.info(f"Train sequences: {len(train_sequences)}")
    logger.info(f"Validation sequences: {len(val_sequences)}")
    logger.info(f"Test sequences: {len(test_sequences)}")
    
    # Calculate maximum word length to use as window size
    max_word_length = max(len(seq) for seq in sequences)
    window_size = max_word_length + 2 # 2 for padding
    logger.info(f"\nCalculated window size from max word length + padding: {window_size}")
    
    # Create CBOW datasets
    logger.info(f"\nCreating CBOW datasets with window size {window_size}...")
    
    # Build global vocabulary from all sequences
    logger.info("Building global vocabulary...")
    vocab = build_vocab(sequences)
    
    # Train dataset
    train_dataset = CBOWDataset(train_sequences, window_size=window_size, vocab=vocab)
    
    # Val and Test datasets (use same global vocab)
    val_dataset = CBOWDataset(val_sequences, window_size=window_size, vocab=vocab)
    test_dataset = CBOWDataset(test_sequences, window_size=window_size, vocab=vocab)
    
    logger.info(f"Vocabulary size: {train_dataset.vocab_size}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create model
    logger.info(f"\nCreating model with embedding dimension {hp.EMBEDDING_DIM}...")
    model = CBOWModel(train_dataset.vocab_size, hp.EMBEDDING_DIM)
    
    # Train
    logger.info(f"\nTraining on {device}...")
    model, losses, val_losses = train_cbow(
        model,
        train_dataset,
        val_dataset=val_dataset,
        epochs=hp.EPOCHS,
        batch_size=hp.BATCH_SIZE,
        learning_rate=hp.LEARNING_RATE,
        patience=hp.PATIENCE,
        min_delta=hp.MIN_DELTA,
        device=device
    )
    
    # Save model and training info
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"cbow_{data_type}_{'_'.join(languages)}_{timestamp}"
    
    # Save model
    model_path = os.path.join(output_dir, f"{model_name}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': train_dataset.vocab_size,
        'embedding_dim': hp.EMBEDDING_DIM,
        'word_to_idx': train_dataset.word_to_idx,
        'idx_to_word': train_dataset.idx_to_word,
    }, model_path)
    logger.info(f"\nModel saved to: {model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump({
            'config': {
                'languages': languages,
                'data_type': data_type,
                'embedding_dim': hp.EMBEDDING_DIM,
                'window_size': window_size,
                'epochs': hp.EPOCHS,
                'batch_size': hp.BATCH_SIZE,
                'learning_rate': hp.LEARNING_RATE,
            },
            'losses': losses,
            'val_losses': val_losses,
            'vocab_size': train_dataset.vocab_size,
            'training_samples': len(train_dataset),
            'validation_samples': len(val_dataset),
            'test_samples': len(test_dataset),
        }, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")
    
    # Evaluate on test set
    if len(test_dataset) > 0:
        logger.info("\nEvaluating on test set...")
        avg_test_loss = evaluate_model(model, test_dataset, batch_size=hp.BATCH_SIZE, device=device)
        logger.info(f"Test Loss: {avg_test_loss:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)
    
    return model_path


if __name__ == "__main__":
    train_model()
