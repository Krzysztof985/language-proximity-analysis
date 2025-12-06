#!/usr/bin/env python3
"""
Test script for CBOW Word2Vec implementation.
"""
import sys
import os

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from src.logger.logging_config import setup_logger
from src.embedding_service.embeding.cbow import CBOWModel, CBOWDataset, train_cbow, find_similar_words

# Set up logger for this module
logger = setup_logger(__name__, 'test_cbow.log')


def test_cbow_simple():
    """Test CBOW with a simple example."""
    logger.info("=" * 50)
    logger.info("Testing CBOW with simple text")
    logger.info("=" * 50)
    
    # Simple text: character-level
    text = "the quick brown fox jumps over the lazy dog"
    tokens = list(text)
    
    logger.info(f"Text: {text}")
    logger.info(f"Number of tokens: {len(tokens)}")
    
    # Create dataset
    window_size = 2
    dataset = CBOWDataset(tokens, window_size=window_size)
    logger.info(f"Vocabulary size: {dataset.vocab_size}")
    logger.info(f"Number of training samples: {len(dataset)}")
    
    # Create model
    embedding_dim = 10
    model = CBOWModel(dataset.vocab_size, embedding_dim)
    logger.info(f"Model created with embedding dimension: {embedding_dim}")
    
    # Train
    logger.info("\nTraining...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model, losses = train_cbow(
        model, 
        dataset, 
        epochs=20, 
        batch_size=16, 
        learning_rate=0.01,
        device=device
    )
    
    # Test similarity
    logger.info("\n" + "=" * 50)
    logger.info("Testing word similarity")
    logger.info("=" * 50)
    
    test_chars = ['t', 'o', 'e', ' ']
    for char in test_chars:
        if char in dataset.word_to_idx:
            similar = find_similar_words(
                model, 
                char, 
                dataset.word_to_idx, 
                dataset.idx_to_word, 
                top_k=3,
                device=device
            )
            logger.info(f"\nMost similar to '{char}':")
            for word, sim in similar:
                logger.info(f"  '{word}': {sim:.4f}")


def test_cbow_with_multilingual_data():
    """Test CBOW with actual multilingual data."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing CBOW with multilingual character data")
    logger.info("=" * 50)
    
    try:
        # Add data directory to path
        data_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        sys.path.insert(0, data_dir_path)
        
        from src.embedding_service.data.datasets.multilingual_dataset import MultilingualWordDataset
        
        # Load data
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
        languages = ["pl"]
        
        word_dataset = MultilingualWordDataset(languages, data_dir)
        logger.info(f"Loaded {len(word_dataset)} characters from {languages}")
        
        # Extract just the characters (ignore language labels for now)
        tokens = [char for char, _ in word_dataset.samples[:10000]]  # Use first 10k for speed
        logger.info(f"Using {len(tokens)} tokens for training")
        
        # Create CBOW dataset
        window_size = 2
        cbow_dataset = CBOWDataset(tokens, window_size=window_size)
        logger.info(f"Vocabulary size: {cbow_dataset.vocab_size}")
        logger.info(f"Training samples: {len(cbow_dataset)}")
        
        # Create and train model
        embedding_dim = 50
        model = CBOWModel(cbow_dataset.vocab_size, embedding_dim)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        logger.info("\nTraining...")
        model, losses = train_cbow(
            model,
            cbow_dataset,
            epochs=5,
            batch_size=128,
            learning_rate=0.001,
            device=device
        )
        
        # Test similarity for Polish characters
        logger.info("\n" + "=" * 50)
        logger.info("Character similarity (Polish)")
        logger.info("=" * 50)
        
        test_chars = ['a', 'e', 'i', 'o', 'ą', 'ę']
        for char in test_chars:
            if char in cbow_dataset.word_to_idx:
                similar = find_similar_words(
                    model,
                    char,
                    cbow_dataset.word_to_idx,
                    cbow_dataset.idx_to_word,
                    top_k=5,
                    device=device
                )
                logger.info(f"\nMost similar to '{char}':")
                for word, sim in similar:
                    logger.info(f"  '{word}': {sim:.4f}")
        
    except Exception as e:
        logger.info(f"Error testing with multilingual data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test 1: Simple example
    test_cbow_simple()
    
    # Test 2: With actual data
    test_cbow_with_multilingual_data()
