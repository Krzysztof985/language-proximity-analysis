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


def train_model():
    parser = argparse.ArgumentParser(description='Train CBOW Word2Vec model')
    
    # Data parameters
    parser.add_argument('--languages', nargs='+', default=['pl'], 
                       help='Languages to train on (default: pl)')
    parser.add_argument('--data-type', choices=['words', 'phonemes'], default='words',
                       help='Type of data to use (default: words)')
    parser.add_argument('--data-dir', type=str, 
                       default='../../data',
                       help='Directory containing language data')
    parser.add_argument('--max-tokens', type=int, default=None,
                       help='Maximum number of tokens to use (default: all)')
    
    # Model parameters
    parser.add_argument('--embedding-dim', type=int, default=100,
                       help='Dimension of embeddings (default: 100)')
    parser.add_argument('--window-size', type=int, default=2,
                       help='Context window size (default: 2)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training (default: auto)')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='../../models',
                       help='Directory to save models (default: ../../models)')
    parser.add_argument('--save-freq', type=int, default=0,
                       help='Save checkpoint every N epochs (0=only final, default: 0)')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, args.data_dir))
    output_dir = os.path.normpath(os.path.join(script_dir, args.output_dir))
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info("=" * 60)
    logger.info("CBOW Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Languages: {args.languages}")
    logger.info(f"Data type: {args.data_type}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Embedding dimension: {args.embedding_dim}")
    logger.info(f"Window size: {args.window_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)
    
    # Load data
    logger.info("\nLoading data...")
    data_path = os.path.join(os.path.dirname(script_dir), "data")
    sys.path.insert(0, data_path)
    
    if args.data_type == 'words':
        from datasets.multilingual_dataset import MultilingualWordDataset
        dataset_class = MultilingualWordDataset
    else:
        from datasets.multilingual_dataset import MultilingualPhonemeDataset
        dataset_class = MultilingualPhonemeDataset
    
    multilang_dataset = dataset_class(args.languages, data_dir)
    logger.info(f"Loaded {len(multilang_dataset)} tokens")
    
    # Extract tokens (ignore language labels for now)
    tokens = [token for token, _ in multilang_dataset.samples]
    if args.max_tokens:
        tokens = tokens[:args.max_tokens]
        logger.info(f"Using first {len(tokens)} tokens")
    
    # Create CBOW dataset
    logger.info(f"\nCreating CBOW dataset with window size {args.window_size}...")
    cbow_dataset = CBOWDataset(tokens, window_size=args.window_size)
    logger.info(f"Vocabulary size: {cbow_dataset.vocab_size}")
    logger.info(f"Training samples: {len(cbow_dataset)}")
    
    # Create model
    logger.info(f"\nCreating model with embedding dimension {args.embedding_dim}...")
    model = CBOWModel(cbow_dataset.vocab_size, args.embedding_dim)
    
    # Train
    logger.info(f"\nTraining on {device}...")
    model, losses = train_cbow(
        model,
        cbow_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # Save model and training info
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"cbow_{args.data_type}_{'_'.join(args.languages)}_{timestamp}"
    
    # Save model
    model_path = os.path.join(output_dir, f"{model_name}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': cbow_dataset.vocab_size,
        'embedding_dim': args.embedding_dim,
        'word_to_idx': cbow_dataset.word_to_idx,
        'idx_to_word': cbow_dataset.idx_to_word,
    }, model_path)
    logger.info(f"\nModel saved to: {model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump({
            'config': vars(args),
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
