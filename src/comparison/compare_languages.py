#!/usr/bin/env python3
"""
Compare multiple language pairs and determine which are most similar.
"""
import os
import sys
import subprocess
import numpy as np
import json
from itertools import combinations

from src.logger.logging_config import setup_logger

logger = setup_logger(__name__, "compare_languages.log")


def run_comparison(lang1, lang2, limit=1000, random_sample=True):
    """Run comparison between two languages."""
    logger.info(f"Comparing {lang1} vs {lang2}...")
    
    # Get project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    cmd = [
        f"{project_root}/.venv/bin/python",
        f"{project_root}/src/comparison/compare_all_words.py",
        lang1, lang2,
        "--limit", str(limit),
        "--batch-size", "500"
    ]
    
    if random_sample:
        cmd.append("--random-sample")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Failed to compare {lang1} vs {lang2}")
        logger.error(result.stderr)
        return None
    
    # Return path to result file
    return f"{project_root}/results/similarity_matrix_{lang1}_{lang2}.npz"


def analyze_similarity(npz_file):
    """Analyze similarity matrix and return mean similarity."""
    data = np.load(npz_file, allow_pickle=True)
    matrix = data['similarity_matrix']
    
    # Calculate mean similarity (excluding diagonal if square matrix)
    if matrix.shape[0] == matrix.shape[1]:
        # Square matrix - exclude diagonal
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        mean_sim = matrix[mask].mean()
    else:
        mean_sim = matrix.mean()
    
    return {
        'mean_similarity': float(mean_sim),
        'std_similarity': float(matrix.std()),
        'max_similarity': float(matrix.max()),
        'min_similarity': float(matrix.min()),
        'shape': matrix.shape
    }


def compare_languages_main():
    # Get project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    languages = ['en', 'pl', 'de', 'fr']
    limit = 1000
    
    print(f"{'='*60}")
    print(f"Comparing language pairs with {limit} random words each")
    print(f"Languages: {', '.join(languages)}")
    print(f"{'='*60}\n")
    
    results = {}
    
    # Generate all pairs
    pairs = list(combinations(languages, 2))
    
    for lang1, lang2 in pairs:
        pair_key = f"{lang1}-{lang2}"
        print(f"\nProcessing {pair_key}...")
        
        # Run comparison
        npz_file = run_comparison(lang1, lang2, limit=limit, random_sample=True)
        
        if npz_file and os.path.exists(npz_file):
            # Analyze results
            stats = analyze_similarity(npz_file)
            results[pair_key] = stats
            
            print(f"  Mean similarity: {stats['mean_similarity']:.4f}")
            print(f"  Std: {stats['std_similarity']:.4f}")
            print(f"  Range: [{stats['min_similarity']:.4f}, {stats['max_similarity']:.4f}]")
        else:
            print(f"  Failed to generate comparison")
    
    # Save summary
    summary_file = f"{project_root}/results/language_similarity_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Language Similarity Ranking")
    print(f"{'='*60}")
    
    # Sort by mean similarity
    sorted_pairs = sorted(results.items(), key=lambda x: x[1]['mean_similarity'], reverse=True)
    
    for rank, (pair, stats) in enumerate(sorted_pairs, 1):
        print(f"{rank}. {pair:10s} - Mean similarity: {stats['mean_similarity']:.4f}")
    
    print(f"\nResults saved to: {summary_file}")


if __name__ == "__main__":
    compare_languages_main()
