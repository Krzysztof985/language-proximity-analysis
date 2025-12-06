#!/usr/bin/env python3
"""
Analyze and query similarity matrix results.
"""
import os
import sys
import argparse
import numpy as np
import json

from src.logger.logging_config import setup_logger

logger = setup_logger(__name__, "analyze_similarity_matrix.log")

def load_similarity_matrix(npz_file):
    """Load similarity matrix from npz file."""
    data = np.load(npz_file, allow_pickle=True)
    return {
        'similarity_matrix': data['similarity_matrix'],
        'words1': data['words1'].tolist(),
        'words2': data['words2'].tolist(),
        'lang1': str(data['lang1']),
        'lang2': str(data['lang2'])
    }


def find_most_similar(data, word, lang, top_k=10):
    """Find most similar words for a given word."""
    if lang == data['lang1']:
        if word not in data['words1']:
            print(f"Word '{word}' not found in {lang}")
            return []
        
        idx = data['words1'].index(word)
        similarities = data['similarity_matrix'][idx]
        target_words = data['words2']
        target_lang = data['lang2']
    elif lang == data['lang2']:
        if word not in data['words2']:
            print(f"Word '{word}' not found in {lang}")
            return []
        
        idx = data['words2'].index(word)
        similarities = data['similarity_matrix'][:, idx]
        target_words = data['words1']
        target_lang = data['lang1']
    else:
        logger.error(f"Language {lang} not in matrix")
        return []
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for i in top_indices:
        results.append((target_words[i], float(similarities[i])))
    
    return results


def print_statistics(data):
    """Print statistics about the similarity matrix."""
    matrix = data['similarity_matrix']
    
    print(f"\n{'='*60}")
    print(f"Similarity Matrix Statistics")
    print(f"{'='*60}")
    print(f"Languages: {data['lang1']} ({len(data['words1'])} words) vs {data['lang2']} ({len(data['words2'])} words)")
    print(f"Matrix shape: {matrix.shape}")
    print(f"\nSimilarity Statistics:")
    print(f"  Mean: {matrix.mean():.4f}")
    print(f"  Std:  {matrix.std():.4f}")
    print(f"  Min:  {matrix.min():.4f}")
    print(f"  Max:  {matrix.max():.4f}")
    
    # Find highest similarity pairs
    print(f"\nTop 10 Most Similar Word Pairs:")
    flat_indices = np.argsort(matrix.flatten())[-10:][::-1]
    for rank, flat_idx in enumerate(flat_indices, 1):
        i, j = np.unravel_index(flat_idx, matrix.shape)
        word1 = data['words1'][i]
        word2 = data['words2'][j]
        sim = matrix[i, j]
        print(f"  {rank}. {word1} ({data['lang1']}) <-> {word2} ({data['lang2']}): {sim:.4f}")


def analyze_similarity_main():
    parser = argparse.ArgumentParser(description='Analyze similarity matrix')
    parser.add_argument('matrix_file', type=str, help='Path to .npz similarity matrix file')
    parser.add_argument('--query', type=str, help='Query word to find similar words')
    parser.add_argument('--lang', type=str, help='Language of query word')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results to show')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    
    args = parser.parse_args()
    
    # Load matrix
    logger.info(f"Loading similarity matrix from {args.matrix_file}...")
    data = load_similarity_matrix(args.matrix_file)
    
    if args.stats:
        print_statistics(data)
    
    if args.query and args.lang:
        print(f"\n{'='*60}")
        print(f"Most similar to '{args.query}' ({args.lang}):")
        print(f"{'='*60}")
        results = find_most_similar(data, args.query, args.lang, args.top_k)
        for rank, (word, sim) in enumerate(results, 1):
            target_lang = data['lang2'] if args.lang == data['lang1'] else data['lang1']
            print(f"  {rank}. {word} ({target_lang}): {sim:.4f}")


if __name__ == "__main__":
    analyze_similarity_main()
