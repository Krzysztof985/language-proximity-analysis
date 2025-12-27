"""
Language Proximity Analysis Backend
Contains core logic for language similarity analysis using different methods.
"""

import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

from src.utils.overall_similarity import add_connection
from src.utils.file_utils import get_words_from_file, save_words_to_file, save_similarity_matrix, load_translations_csv, save_translations_csv
from src.utils.translate import translate_words
from src.utils.similarity import compute_similarity
from src.utils.overall_similarity import diagonal_average
from src.embedding_service.compare_words import WordComparator
from src.logger.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__, "analysis_backend.log")


def compute_similarity_levenshtein(word1: str, word2: str) -> float:
    """Compute similarity using Levenshtein distance"""
    return compute_similarity(word1, word2)


class OptimizedWordComparator:
    """Optimized wrapper for WordComparator with cached phoneme dictionaries"""
    
    def __init__(self, languages: List[str], data_dir: str = 'data'):
        self.comparator = WordComparator(languages=languages, data_dir=data_dir)
        self.phoneme_dicts = {}
        self.data_dir = data_dir
        
        # Pre-load phoneme dictionaries for all languages
        for lang in languages:
            self.phoneme_dicts[lang] = self.comparator.load_phoneme_dictionary(lang, data_dir)
            
    def compare_words(self, word1: str, lang1: str, word2: str, lang2: str) -> Dict:
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


def compute_similarity_embedding(word1: str, lang1: str, word2: str, lang2: str, comparator: OptimizedWordComparator) -> float:
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


def run_levenshtein_analysis(
    languages: List[str], 
    data_dir: str, 
    results_dir: str, 
    base_language: str = "en"
) -> None:
    """
    Run language proximity analysis using Levenshtein distance method.
    
    Args:
        languages: List of language codes
        data_dir: Path to data directory
        results_dir: Path to results directory
        base_language: Base language code for translations
    """
    logger.info("Starting language proximity analysis using Levenshtein method")
    
    method_suffix = "_levenshtein"
    _create_output_directories(results_dir, method_suffix)
    
    for filename in os.listdir(data_dir):
        if not filename.endswith(".txt"):
            continue
            
        topic = filename.replace(".txt", "")
        print(f"=== Processing topic: {topic} (Levenshtein) ===")
        
        words = get_words_from_file(os.path.join(data_dir, filename))
        print(f"Loaded {len(words)} words from {filename}")
        
        # Load existing translations or create new ones (CSV format)
        translations = _load_or_create_translations_csv(words, languages, base_language, 
                                                       results_dir, topic)
        
        # Compute similarities and create graphs
        _process_language_pairs(
            translations, languages, results_dir, method_suffix, topic, 
            method="levenshtein"
        )
        
    logger.info("Levenshtein analysis completed successfully!")
    print(f"Levenshtein analysis completed! Check results folder with suffix '{method_suffix}'")


def run_embedding_analysis(
    languages: List[str], 
    data_dir: str, 
    results_dir: str, 
    base_language: str = "en"
) -> None:
    """
    Run language proximity analysis using embedding method.
    
    Args:
        languages: List of language codes
        data_dir: Path to data directory
        results_dir: Path to results directory
        base_language: Base language code for translations
    """
    logger.info("Starting language proximity analysis using embedding method")
    
    method_suffix = "_embedding"
    _create_output_directories(results_dir, method_suffix)
    
    # Initialize embedding comparator
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
        raise RuntimeError(f"Embedding analysis failed: {e}")
    
    for filename in os.listdir(data_dir):
        if not filename.endswith(".txt"):
            continue
            
        topic = filename.replace(".txt", "")
        print(f"=== Processing topic: {topic} (Embedding) ===")
        
        words = get_words_from_file(os.path.join(data_dir, filename))
        print(f"Loaded {len(words)} words from {filename}")
        
        # Load existing translations or create new ones (CSV format)
        translations = _load_or_create_translations_csv(words, languages, base_language, 
                                                       results_dir, topic)
        
        # Compute similarities and create graphs
        _process_language_pairs(
            translations, languages, results_dir, method_suffix, topic, 
            method="embedding", comparator=comparator
        )
        
    logger.info("Embedding analysis completed successfully!")
    print(f"Embedding analysis completed! Check results folder with suffix '{method_suffix}'")


def _load_or_create_translations_csv(words: List[str], languages: List[str], base_language: str, 
                                     results_dir: str, topic: str) -> Dict[str, List[str]]:
    """Load existing translations from CSV or create new ones if needed."""
    translations_dir = f"{results_dir}/translations"
    csv_path = f"{translations_dir}/{topic}.csv"
    
    # Try to load existing CSV
    translations = load_translations_csv(csv_path)
    
    if translations:
        logger.info(f"Loaded existing translations from {csv_path}")
        
        # Check if we have all required languages and add missing ones
        missing_languages = set(languages) - set(translations.keys())
        if missing_languages:
            for lang in missing_languages:
                if lang == base_language:
                    translations[lang] = words
                    logger.info(f"Added base language {lang}")
                else:
                    logger.info(f"Creating new translations for {lang}")
                    translations[lang] = translate_words(words, lang)
            
            # Save updated CSV
            save_translations_csv(translations, csv_path)
            logger.info(f"Updated translations saved to {csv_path}")
    else:
        # Create new translations for all languages
        logger.info(f"Creating new translation file: {csv_path}")
        translations = {}
        
        for lang in languages:
            if lang == base_language:
                translations[lang] = words
            else:
                logger.info(f"Creating new translations for {lang}")
                translations[lang] = translate_words(words, lang)
        
        # Save new CSV
        save_translations_csv(translations, csv_path)
        logger.info(f"New translations saved to {csv_path}")
    
    return translations


def _create_output_directories(results_dir: str, method_suffix: str) -> None:
    """Create output directories for the analysis."""
    os.makedirs(f"{results_dir}/translations", exist_ok=True)  # Unified translations folder
    os.makedirs(f"{results_dir}/similarities{method_suffix}", exist_ok=True)
    os.makedirs(f"{results_dir}/graphs{method_suffix}", exist_ok=True)



def _process_language_pairs(
    translations: Dict[str, List[str]], 
    languages: List[str], 
    results_dir: str, 
    method_suffix: str, 
    topic: str, 
    method: str,
    comparator: Optional[OptimizedWordComparator] = None
) -> None:
    """Process all language pairs and create similarity matrices and graphs."""
    
    G = nx.Graph()  # Create graph for this topic
    
    # Compute similarities between language pairs
    logger.info(f"Computing similarities for topic '{topic}' using {method} method")
    
    for lang_idx1 in range(len(languages)):
        for lang_idx2 in range(lang_idx1 + 1, len(languages)):
            lang1, lang2 = languages[lang_idx1], languages[lang_idx2]
            logger.info(f"Processing language pair: {lang1} -> {lang2}")
            
            # Compute similarity matrix
            matrix = _compute_similarity_matrix(
                translations[lang1], lang1, translations[lang2], lang2, method, comparator
            )
            
            # Save similarity matrix
            save_similarity_matrix(
                translations[lang1], translations[lang2], matrix,
                f"{results_dir}/similarities{method_suffix}/{topic}_{lang1}_{lang2}.csv"
            )
            
            # Add connection to graph
            outcome = diagonal_average(matrix) * 100
            add_connection(G, lang1, lang2, f"{round(outcome, 2)}%")
    
    # Create and save graph
    _create_and_save_graph(G, topic, method, results_dir, method_suffix)


def _compute_similarity_matrix(
    words1: List[str], lang1: str, words2: List[str], lang2: str, 
    method: str, comparator: Optional[OptimizedWordComparator] = None
) -> List[List[float]]:
    """Compute similarity matrix between two lists of words."""
    
    matrix = []
    total_comparisons = len(words1) * len(words2)
    completed = 0
    
    logger.info(f"Computing {total_comparisons} word comparisons for {lang1}-{lang2}")
    
    for word1_idx, w1 in enumerate(words1):
        row = []
        for word2_idx, w2 in enumerate(words2):
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
    
    return matrix


def _create_and_save_graph(G: nx.Graph, topic: str, method: str, results_dir: str, method_suffix: str) -> None:
    """Create and save similarity graph."""
    pos = nx.spiral_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=700)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=nx.get_edge_attributes(G, "label"), 
        font_size=5, label_pos=0.6
    )
    
    plt.title(f"{topic} related words similarity ({method} method)")
    plt.savefig(
        f"{results_dir}/graphs{method_suffix}/{topic}_similarity_graph.png", 
        format="png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    logger.info(f"Graph saved for topic '{topic}'")