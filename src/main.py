"""
Main entry point for Language Proximity Analysis
Simple CLI interface that delegates to backend functions.
"""

import os
import sys
import argparse


from src.analysis_backend import run_levenshtein_analysis, run_embedding_analysis
from src.logger.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__, "main.log")

# Config
BASE_LANGUAGE = "en"  
languages = ["fi", "pt", "pl", "es", "en", "fr", "it", "nl", "sv", "sl"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "../data")
results_dir = os.path.join(BASE_DIR, "../results")


def main(method: str = "levenshtein") -> None:
    """
    Main function for language proximity analysis
    
    Args:
        method: Comparison method ('levenshtein' or 'embedding')
    """
    print(f"Starting Language Proximity Analysis using {method.upper()} method")
    print(f"Languages: {', '.join(languages)}")
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print("-" * 60)
    
    try:
        if method == "embedding":
            run_embedding_analysis(
                languages=languages,
                data_dir=data_dir,
                results_dir=results_dir,
                base_language=BASE_LANGUAGE
            )
        elif method == "levenshtein":
            run_levenshtein_analysis(
                languages=languages,
                data_dir=data_dir,
                results_dir=results_dir,
                base_language=BASE_LANGUAGE
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'levenshtein' or 'embedding'")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\nAnalysis failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Language Proximity Analysis - Compare language similarity using different methods'
    )
    parser.add_argument(
        '--method', 
        choices=['levenshtein', 'embedding'], 
        default='levenshtein',
        help='Comparison method: levenshtein (default, fast) or embedding (advanced, slower)'
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        default=languages,
        help=f'Languages to analyze (default: {" ".join(languages)})'
    )
    
    args = parser.parse_args()
    
    # Override languages if provided
    if args.languages != languages:
        languages = args.languages
        print(f"Using custom languages: {', '.join(languages)}")
    
    try:
        main(method=args.method)
        print(f"Analysis completed successfully using {args.method.upper()} method!")
    except KeyboardInterrupt:
        print("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)


