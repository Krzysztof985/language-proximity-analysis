#!/usr/bin/env python3
"""
Unified runner script for language similarity comparisons.
"""
import argparse
import sys
import os

from src.comparison.compare_all_words import (
    load_all_words, 
    compute_embeddings_batch,
    compute_similarity_matrix,
    save_results
)
from src.embedding_service.compare_words import WordComparator
from src.logger.logging_config import setup_logger

logger = setup_logger(__name__, "run_comparison.log")


def run_comparison(lang1, lang2, limit=None, random_sample=False, batch_size=1000, 
                   data_dir='data', output_dir='results'):
    """
    Run a comparison between two languages.
    
    Args:
        lang1: First language code
        lang2: Second language code
        limit: Optional limit on number of words
        random_sample: Whether to randomly sample words
        batch_size: Batch size for processing
        data_dir: Data directory path
        output_dir: Output directory path
    
    Returns:
        Path to output file
    """
    # Get project root (src/ is one level down from root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    data_dir = os.path.join(project_root, data_dir)
    output_dir = os.path.join(project_root, output_dir)
    
    logger.info(f"Comparing {lang1} vs {lang2}")
    logger.info(f"Limit: {limit}, Random sample: {random_sample}")
    
    # Find models
    models_dir = os.path.join(project_root, 'models')
    phoneme_model = None
    word_model = None
    
    for f in os.listdir(models_dir):
        if f.startswith('cbow_phonemes_') and f.endswith('.pt') and lang1 in f and lang2 in f:
            phoneme_model = os.path.join(models_dir, f)
        if f.startswith('cbow_words_') and f.endswith('.pt') and lang1 in f and lang2 in f:
            word_model = os.path.join(models_dir, f)
    
    if not phoneme_model or not word_model:
        logger.error(f"Could not find models for {lang1} and {lang2}")
        return None
    
    logger.info(f"Using phoneme model: {os.path.basename(phoneme_model)}")
    logger.info(f"Using word model: {os.path.basename(word_model)}")
    
    # Initialize comparator
    comparator = WordComparator(phoneme_model, word_model)
    
    # Load words
    words1, phonemes1 = load_all_words(comparator, lang1, data_dir, limit, random_sample)
    words2, phonemes2 = load_all_words(comparator, lang2, data_dir, limit, random_sample)
    
    # Compute embeddings
    embeddings1 = compute_embeddings_batch(comparator, words1, phonemes1, batch_size)
    embeddings2 = compute_embeddings_batch(comparator, words2, phonemes2, batch_size)
    
    # Compute similarity matrix
    similarity_matrix, temp_filename = compute_similarity_matrix(embeddings1, embeddings2, batch_size)
    
    # Save results
    output_file = save_results(words1, words2, similarity_matrix, lang1, lang2, output_dir, temp_filename)
    
    logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
    logger.info(f"Results saved to: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Run language similarity comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two languages with 1000 random words
  python src/run_comparison.py pl en --limit 1000 --random-sample
  
  # Compare multiple languages (all pairs)
  python src/run_comparison.py pl en fr es --limit 1000 --random-sample
  
  # Full comparison (warning: slow!)
  python src/run_comparison.py pl en
        """
    )
    
    parser.add_argument('languages', nargs='+', type=str, 
                       help='Language codes to compare (e.g., pl en fr). If more than 2, compares all pairs.')
    parser.add_argument('--limit', type=int, default=None, 
                       help='Limit number of words per language')
    parser.add_argument('--random-sample', action='store_true',
                       help='Randomly sample words when using --limit')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for processing (default: 1000)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory (default: results)')
    
    args = parser.parse_args()
    
    # Generate language pairs
    from itertools import combinations
    
    if len(args.languages) < 2:
        logger.error("Need at least 2 languages to compare")
        print("Error: Please provide at least 2 language codes")
        sys.exit(1)
    
    if len(args.languages) == 2:
        pairs = [(args.languages[0], args.languages[1])]
    else:
        pairs = list(combinations(args.languages, 2))
        logger.info(f"Comparing {len(pairs)} language pairs from {len(args.languages)} languages")
    
    results = {}
    
    for lang1, lang2 in pairs:
        print(f"\n{'='*60}")
        print(f"Comparing {lang1} vs {lang2}")
        print(f"{'='*60}")
        
        output_file = run_comparison(
            lang1, lang2,
            limit=args.limit,
            random_sample=args.random_sample,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
        
        if output_file:
            results[f"{lang1}-{lang2}"] = output_file
            print(f"✓ Results saved to: {output_file}")
        else:
            print(f"✗ Comparison failed. Check logs for details.")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Completed {len(results)}/{len(pairs)} comparisons")
    
    if results:
        print("\nGenerated files:")
        for pair, file in results.items():
            print(f"  {pair}: {file}")
        
        print(f"\nTo analyze results:")
        first_file = list(results.values())[0]
        print(f"  PYTHONPATH=$(pwd) .venv/bin/python src/comparison/analyze_similarity_matrix.py {first_file} --stats")
    
    if len(results) < len(pairs):
        sys.exit(1)


if __name__ == "__main__":
    main()
