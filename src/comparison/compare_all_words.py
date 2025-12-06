#!/usr/bin/env python3
"""
Compare all words from one language with all words from another language.
Generates a full similarity matrix.
"""
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
import json

from src.embedding_service.compare_words import WordComparator
from src.logger.logging_config import setup_logger

logger = setup_logger(__name__, "compare_all_words.log")


def load_all_words(comparator, lang, data_dir, limit=None, random_sample=False):
    """Load all words and their phonemes for a language."""
    phoneme_dict = comparator.load_phoneme_dictionary(lang, data_dir)
    
    words = []
    phonemes_list = []
    
    # Convert to list for random sampling
    items = list(phoneme_dict.items())
    
    # Random sampling if requested
    if limit and random_sample and len(items) > limit:
        import random
        random.seed(42)  # For reproducibility
        items = random.sample(items, limit)
        logger.info(f"Randomly sampled {limit} words from {len(phoneme_dict)} total words for {lang}")
    
    for word, phonemes in items:
        words.append(word)
        phonemes_list.append(phonemes)
        if limit and not random_sample and len(words) >= limit:
            break
    
    logger.info(f"Loaded {len(words)} words for {lang}")
    return words, phonemes_list


def compute_embeddings_batch(comparator, words, phonemes_list, batch_size=1000):
    """Compute embeddings for all words in batches."""
    all_embeddings = []
    
    logger.info(f"Computing embeddings for {len(words)} words...")
    for i in tqdm(range(0, len(words), batch_size), desc="Computing embeddings"):
        batch_words = words[i:i+batch_size]
        batch_phonemes = phonemes_list[i:i+batch_size]
        
        batch_embeddings = []
        for word, phonemes in zip(batch_words, batch_phonemes):
            emb = comparator.get_combined_embedding(word, phonemes)
            batch_embeddings.append(emb)
        
        all_embeddings.extend(batch_embeddings)
    
    return np.array(all_embeddings)


def compute_similarity_matrix(embeddings1, embeddings2, batch_size=1000):
    """Compute cosine similarity matrix between two sets of embeddings."""
    n1 = len(embeddings1)
    n2 = len(embeddings2)
    
    logger.info(f"Computing similarity matrix: {n1} x {n2}")
    
    # Normalize embeddings for cosine similarity
    embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    # Compute in batches to manage memory
    similarity_matrix = np.zeros((n1, n2), dtype=np.float32)
    
    for i in tqdm(range(0, n1, batch_size), desc="Computing similarities"):
        end_i = min(i + batch_size, n1)
        batch1 = embeddings1_norm[i:end_i]
        
        # Compute similarity for this batch against all of embeddings2
        similarity_matrix[i:end_i] = np.dot(batch1, embeddings2_norm.T)
    
    return similarity_matrix


def save_results(words1, words2, similarity_matrix, lang1, lang2, output_dir, format='npz'):
    """Save similarity matrix and word lists."""
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = f"similarity_matrix_{lang1}_{lang2}"
    
    if format == 'npz':
        # Save as compressed numpy array (most efficient)
        output_file = os.path.join(output_dir, f"{base_name}.npz")
        np.savez_compressed(
            output_file,
            similarity_matrix=similarity_matrix,
            words1=words1,
            words2=words2,
            lang1=lang1,
            lang2=lang2
        )
        logger.info(f"Saved similarity matrix to {output_file}")
        
        # Also save metadata as JSON
        metadata_file = os.path.join(output_dir, f"{base_name}_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'lang1': lang1,
                'lang2': lang2,
                'num_words1': len(words1),
                'num_words2': len(words2),
                'matrix_shape': similarity_matrix.shape,
                'format': 'npz'
            }, f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")
    
    return output_file


def compare_all_words_main():
    parser = argparse.ArgumentParser(description='Compare all words between two languages')
    parser.add_argument('lang1', type=str, help='First language code (e.g., pl)')
    parser.add_argument('lang2', type=str, help='Second language code (e.g., en)')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of words per language (for testing)')
    parser.add_argument('--random-sample', action='store_true', help='Randomly sample words when using --limit')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, args.data_dir)
    output_dir = os.path.join(project_root, args.output_dir)
    
    # Find models
    models_dir = os.path.join(project_root, 'models')
    phoneme_model = None
    word_model = None
    
    for f in os.listdir(models_dir):
        if f.startswith('cbow_phonemes_') and f.endswith('.pt') and args.lang1 in f and args.lang2 in f:
            phoneme_model = os.path.join(models_dir, f)
        if f.startswith('cbow_words_') and f.endswith('.pt') and args.lang1 in f and args.lang2 in f:
            word_model = os.path.join(models_dir, f)
    
    if not phoneme_model or not word_model:
        logger.error(f"Could not find models for {args.lang1} and {args.lang2}")
        return
    
    logger.info(f"Using phoneme model: {os.path.basename(phoneme_model)}")
    logger.info(f"Using word model: {os.path.basename(word_model)}")
    
    # Initialize comparator
    comparator = WordComparator(phoneme_model, word_model)
    
    # Load words
    words1, phonemes1 = load_all_words(comparator, args.lang1, data_dir, args.limit, args.random_sample)
    words2, phonemes2 = load_all_words(comparator, args.lang2, data_dir, args.limit, args.random_sample)
    
    # Compute embeddings
    embeddings1 = compute_embeddings_batch(comparator, words1, phonemes1, args.batch_size)
    embeddings2 = compute_embeddings_batch(comparator, words2, phonemes2, args.batch_size)
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings1, embeddings2, args.batch_size)
    
    # Save results
    output_file = save_results(words1, words2, similarity_matrix, args.lang1, args.lang2, output_dir)
    
    logger.info("Done!")
    logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    compare_all_words_main()
