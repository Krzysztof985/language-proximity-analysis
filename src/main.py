import os
import sys
import argparse
import networkx as nx
import matplotlib.pyplot as plt

from src.utils.overall_similarity import add_connection
from src.utils.file_utils import get_words_from_file, save_words_to_file, save_similarity_matrix
from src.utils.translate import translate_words
from src.utils.similarity import compute_similarity
from src.utils.overall_similarity import diagonal_average
from src.embedding_service.compare_words import WordComparator
from src.logger.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__, "main.log")

# Config
BASE_LANGUAGE = "en"  
# languages = ["en", "pl", "es", "fr", "de", "pt", "it", "sl", "sk", "sv"]
languages = ["en", "pl", "es"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "../data")
results_dir = os.path.join(BASE_DIR, "../results")


def compute_similarity_levenshtein(word1, word2):
    """Compute similarity using Levenshtein distance"""
    return compute_similarity(word1, word2)


class OptimizedWordComparator:
    """Optimized wrapper for WordComparator with cached phoneme dictionaries"""
    
    def __init__(self, languages, data_dir='data'):
        self.comparator = WordComparator(languages=languages, data_dir=data_dir)
        self.phoneme_dicts = {}
        self.data_dir = data_dir
        
        # Pre-load phoneme dictionaries for all languages
        for lang in languages:
            self.phoneme_dicts[lang] = self.comparator.load_phoneme_dictionary(lang, data_dir)
            
    def compare_words(self, word1, lang1, word2, lang2):
        """Compare two words using cached phoneme dictionaries"""
        try:
            # Get phonemes from cached dictionaries
            phonemes1 = self.phoneme_dicts.get(lang1, {}).get(word1)
            phonemes2 = self.phoneme_dicts.get(lang2, {}).get(word2)
            
            if phonemes1 is None or phonemes2 is None:
                return {'cosine_similarity': 0.0}
                
            # Use the comparator's phoneme comparison method directly
            results = self.comparator.compare_phoneme_sequences(word1, phonemes1, word2, phonemes2)
            return results
            
        except Exception as e:
            return {'cosine_similarity': 0.0}


def compute_similarity_embedding(word1, lang1, word2, lang2, comparator):
    """Compute similarity using embedding-based comparison"""
    try:
        result = comparator.compare_words(word1, lang1, word2, lang2)
        if result and 'cosine_similarity' in result:
            # Convert cosine similarity (range [-1, 1]) to normalized similarity (range [0, 1])
            cosine_sim = result['cosine_similarity']
            normalized_sim = (cosine_sim + 1) / 2  # Convert from [-1,1] to [0,1]
            return normalized_sim
        else:
            return 0.0
    except Exception as e:
        return 0.0

def main(method="levenshtein"):
    """Main funCcction for language proximity analysis
    
    Args:
        method: Comparison method ('levenshtein' or 'embedding')
    """
    logger.info(f"Starting language proximity analysis using {method} method")
    
    # Create output directories with method suffix
    method_suffix = f"_{method}"
    os.makedirs(f"{results_dir}/translations{method_suffix}", exist_ok=True)
    os.makedirs(f"{results_dir}/similarities{method_suffix}", exist_ok=True)
    os.makedirs(f"{results_dir}/graphs{method_suffix}", exist_ok=True)
    
    # Initialize embedding comparator if needed
    comparator = None
    if method == "embedding":
        try:
            logger.info("Initializing embedding-based comparator...")
            
            # Temporarily modify sys.argv to pass languages to data pipeline
            original_argv = sys.argv[:]
            sys.argv = ['main.py'] + languages
            
            comparator = OptimizedWordComparator(languages=languages, data_dir=data_dir)
            
            # Restore original sys.argv
            sys.argv = original_argv
            
            logger.info("Embedding comparator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding comparator: {e}")
            logger.info("Falling back to Levenshtein method")
            method = "levenshtein"
            method_suffix = "_levenshtein"

    for filename in os.listdir(data_dir):
        if not filename.endswith(".txt"):
            continue

        topic = filename.replace(".txt", "")
        print(f"\n=== Processing topic: {topic} ===")
        G = nx.Graph() # Graph will be stored here
        words = get_words_from_file(os.path.join(data_dir, filename))
        print(f"Loaded {len(words)} words from {filename}")

        translations = {}
        for lang in languages:
            if lang == BASE_LANGUAGE:
                translations[lang] = words
            else:
                translations[lang] = translate_words(words, lang)

        # Save translations
        for lang, trans_words in translations.items():
            save_words_to_file(trans_words, f"{results_dir}/translations{method_suffix}/{topic}_{lang}.txt")

        # Compute similarities between language pairs
        logger.info(f"Computing similarities for topic '{topic}' using {method} method")
        for lang_idx1 in range(len(languages)):
            for lang_idx2 in range(lang_idx1 + 1, len(languages)):
                lang1, lang2 = languages[lang_idx1], languages[lang_idx2]
                logger.info(f"Processing language pair: {lang1} -> {lang2}")
                
                # Compute similarity matrix based on method
                matrix = []
                total_comparisons = len(translations[lang1]) * len(translations[lang2])
                completed = 0
                
                logger.info(f"Computing {total_comparisons} word comparisons for {lang1}-{lang2}")
                
                for word1_idx, w1 in enumerate(translations[lang1]):
                    row = []
                    for word2_idx, w2 in enumerate(translations[lang2]):
                        try:
                            if method == "embedding" and comparator is not None:
                                sim = compute_similarity_embedding(w1, lang1, w2, lang2, comparator)
                            else:
                                sim = compute_similarity_levenshtein(w1, w2)
                            row.append(sim)
                        except Exception as e:
                            logger.error(f"Error comparing '{w1}' vs '{w2}': {e}")
                            row.append(0.0)
                        
                        completed += 1
                        
                        # Print progress every 10% for large matrices
                        if total_comparisons > 100 and completed % (total_comparisons // 10) == 0:
                            progress = (completed / total_comparisons) * 100
                            logger.info(f"Progress: {progress:.0f}% ({completed}/{total_comparisons}) for {lang1}-{lang2}")
                    
                    matrix.append(row)
                
                # Save similarity matrix
                save_similarity_matrix(translations[lang1], translations[lang2], matrix,
                                       f"{results_dir}/similarities{method_suffix}/{topic}_{lang1}_{lang2}.csv")
                
                # Add connection to graph
                outcome = diagonal_average(matrix) * 100
                add_connection(G, lang1, lang2, f"{round(outcome, 2)}%")
        
        # Create and save graph
        pos = nx.spiral_layout(G)
        nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=700)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"), font_size=5, label_pos=0.6)
        
        plt.title(f"{topic} related words similarity ({method} method)")
        plt.savefig(f"{results_dir}/graphs{method_suffix}/{topic}_similarity_graph.png", format="png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Graph saved for topic '{topic}'")


    logger.info(f"All topics processed successfully using {method} method!")
    print(f"\n✅ All topics processed successfully using {method} method!")
    print(f"Check results folder with suffix '{method_suffix}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Language Proximity Analysis')
    parser.add_argument(
        '--method', 
        choices=['levenshtein', 'embedding'], 
        default='levenshtein',
        help='Comparison method: levenshtein (default) or embedding'
    )
    
    args = parser.parse_args()
    
    try:
        main(method=args.method)
    except KeyboardInterrupt:
        print("\n Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n❌ Analysis failed: {e}")
        sys.exit(1)
