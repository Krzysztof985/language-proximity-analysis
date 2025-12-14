"""
Module for comparing words across languages using trained CBOW embeddings.
"""
import os
import sys
import torch
import numpy as np
from typing import Tuple, List, Dict


from src.logger.logging_config import setup_logger
from src.embedding_service.embeding.cbow import CBOWModel

# Set up logger
logger = setup_logger(__name__, "word_comparison.log")


class WordComparator:
    """
    Compare words across languages using CBOW embeddings.
    """
    
    def __init__(self, phoneme_model_path: str, word_model_path: str, device: str = 'cpu'):
        """
        Initialize the comparator with trained phoneme and word models.
        
        Args:
            phoneme_model_path: Path to the trained phoneme model (.pt file)
            word_model_path: Path to the trained word model (.pt file)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = device
        
        # Load phoneme model
        logger.info(f"Loading phoneme model from {phoneme_model_path}")
        phoneme_checkpoint = torch.load(phoneme_model_path, map_location=device)
        
        self.phoneme_vocab_size = phoneme_checkpoint['vocab_size']
        self.phoneme_embedding_dim = phoneme_checkpoint['embedding_dim']
        self.phoneme_to_idx = phoneme_checkpoint['word_to_idx']
        self.idx_to_phoneme = phoneme_checkpoint['idx_to_word']
        
        self.phoneme_model = CBOWModel(self.phoneme_vocab_size, self.phoneme_embedding_dim)
        self.phoneme_model.load_state_dict(phoneme_checkpoint['model_state_dict'])
        self.phoneme_model.to(device)
        self.phoneme_model.eval()
        
        logger.info(f"Phoneme model loaded. Vocab size: {self.phoneme_vocab_size}, Embedding dim: {self.phoneme_embedding_dim}")
        
        # Load word model
        logger.info(f"Loading word model from {word_model_path}")
        word_checkpoint = torch.load(word_model_path, map_location=device)
        
        self.word_vocab_size = word_checkpoint['vocab_size']
        self.word_embedding_dim = word_checkpoint['embedding_dim']
        self.char_to_idx = word_checkpoint['word_to_idx']
        self.idx_to_char = word_checkpoint['idx_to_word']
        
        self.word_model = CBOWModel(self.word_vocab_size, self.word_embedding_dim)
        self.word_model.load_state_dict(word_checkpoint['model_state_dict'])
        self.word_model.to(device)
        self.word_model.eval()
        
        logger.info(f"Word model loaded. Vocab size: {self.word_vocab_size}, Embedding dim: {self.word_embedding_dim}")
        
        # Combined embedding dimension
        self.combined_embedding_dim = self.phoneme_embedding_dim + self.word_embedding_dim
        logger.info(f"Combined embedding dimension: {self.combined_embedding_dim}")
    
    def phonemes_to_embedding(self, phonemes: List[str]) -> np.ndarray:
        """
        Convert a list of phonemes to an average embedding vector.
        
        Args:
            phonemes: List of phoneme characters (IPA)
            
        Returns:
            Average embedding vector
        """
        embeddings = []
        unk_idx = self.phoneme_to_idx.get('<UNK>')
        
        for phoneme in phonemes:
            if phoneme in self.phoneme_to_idx:
                idx = self.phoneme_to_idx[phoneme]
            elif unk_idx is not None:
                idx = unk_idx
                logger.warning(f"Phoneme '{phoneme}' not in vocabulary, using <UNK>")
            else:
                logger.warning(f"Phoneme '{phoneme}' not in vocabulary, skipping")
                continue
            
            # Get embedding
            idx_tensor = torch.tensor([idx], dtype=torch.long).to(self.device)
            with torch.no_grad():
                embedding = self.phoneme_model.embeddings(idx_tensor).cpu().numpy()[0]
            embeddings.append(embedding)
        
        if not embeddings:
            logger.warning("No valid phonemes found, returning zero vector")
            return np.zeros(self.phoneme_embedding_dim)
        
        # Average all phoneme embeddings
        return np.mean(embeddings, axis=0)
    
    def characters_to_embedding(self, characters: List[str]) -> np.ndarray:
        """
        Convert a list of characters to an average embedding vector.
        
        Args:
            characters: List of characters
            
        Returns:
            Average embedding vector
        """
        embeddings = []
        unk_idx = self.char_to_idx.get('<UNK>')
        
        for char in characters:
            if char in self.char_to_idx:
                idx = self.char_to_idx[char]
            elif unk_idx is not None:
                idx = unk_idx
                logger.warning(f"Character '{char}' not in vocabulary, using <UNK>")
            else:
                logger.warning(f"Character '{char}' not in vocabulary, skipping")
                continue
            
            # Get embedding
            idx_tensor = torch.tensor([idx], dtype=torch.long).to(self.device)
            with torch.no_grad():
                embedding = self.word_model.embeddings(idx_tensor).cpu().numpy()[0]
            embeddings.append(embedding)
        
        if not embeddings:
            logger.warning("No valid characters found, returning zero vector")
            return np.zeros(self.word_embedding_dim)
        
        # Average all character embeddings
        return np.mean(embeddings, axis=0)
    
    def get_combined_embedding(self, word: str, phonemes: List[str]) -> np.ndarray:
        """
        Get combined embedding by concatenating phoneme and character embeddings.
        
        Args:
            word: The word as a string
            phonemes: List of phoneme characters (IPA)
            
        Returns:
            Concatenated embedding vector
        """
        # Get phoneme embedding
        phoneme_emb = self.phonemes_to_embedding(phonemes)
        
        # Get character embedding
        characters = list(word)
        char_emb = self.characters_to_embedding(characters)
        
        # Concatenate
        combined = np.concatenate([phoneme_emb, char_emb])
        
        return combined
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def compare_phoneme_sequences(self, word1: str, phonemes1: List[str], word2: str, phonemes2: List[str]) -> Dict[str, float]:
        """
        Compare two words using combined phoneme and character embeddings.
        
        Args:
            word1: First word as string
            phonemes1: First phoneme sequence
            word2: Second word as string
            phonemes2: Second phoneme sequence
            
        Returns:
            Dictionary with similarity metrics
        """
        # Get combined embeddings
        emb1 = self.get_combined_embedding(word1, phonemes1)
        emb2 = self.get_combined_embedding(word2, phonemes2)
        
        # Cosine similarity
        similarity = self.cosine_similarity(emb1, emb2)
        
        # Euclidean distance
        euclidean_dist = np.linalg.norm(emb1 - emb2)
        
        return {
            'cosine_similarity': float(similarity),
            'euclidean_distance': float(euclidean_dist),
            'phonemes1_length': len(phonemes1),
            'phonemes2_length': len(phonemes2)
        }
    
    def load_phoneme_dictionary(self, lang: str, data_dir: str) -> Dict[str, List[str]]:
        """
        Load phoneme dictionary for a language.
        
        Args:
            lang: Language code (e.g., 'pl', 'en')
            data_dir: Data directory path
            
        Returns:
            Dictionary mapping words to phoneme lists
        """
        phoneme_dict = {}
        file_path = os.path.join(data_dir, lang, "phonemes.txt")
        
        if not os.path.exists(file_path):
            logger.warning(f"Phoneme file not found for {lang} at {file_path}")
            return phoneme_dict
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    word = parts[0]
                    ipa_str = parts[1]
                    phonemes = ipa_str.split()
                    phoneme_dict[word] = phonemes
        
        logger.info(f"Loaded {len(phoneme_dict)} words for {lang}")
        return phoneme_dict
    
    
    def find_models_for_languages(self, lang1: str, lang2: str, data_dir: str) -> Tuple[str, str]:
        """
        Find phoneme and word models for the given language pair.
        Models can contain more languages than requested (superset).
        
        Args:
            lang1: First language code
            lang2: Second language code
            data_dir: Data directory (used to locate models dir)
            
        Returns:
            Tuple of (phoneme_model_path, word_model_path) or None if not found
        """
        # Assuming models are in a 'models' directory at the same level as 'data'
        if data_dir.endswith('data'):
            models_dir = os.path.join(os.path.dirname(data_dir), 'models')
        else:
            models_dir = os.path.join(data_dir, '../models')
            
        models_dir = os.path.normpath(models_dir)
        
        if not os.path.exists(models_dir):
            logger.warning(f"Models directory not found at {models_dir}")
            return None
            
        required_langs = {lang1, lang2}
        
        # Find all potential model files
        candidates = []
        for filename in os.listdir(models_dir):
            if not filename.endswith('.pt'):
                continue
                
            # Parse filename: cbow_{type}_{langs}.pt
            # type is 'phonemes' or 'words'
            if filename.startswith('cbow_phonemes_'):
                model_type = 'phonemes'
                prefix_len = len('cbow_phonemes_')
            elif filename.startswith('cbow_words_'):
                model_type = 'words'
                prefix_len = len('cbow_words_')
            else:
                continue
                
            # Extract languages part
            langs_part = filename[prefix_len:-3] # remove prefix and .pt
            model_langs = set(langs_part.split('_'))
            
            # Check if model contains required languages
            if required_langs.issubset(model_langs):
                candidates.append({
                    'filename': filename,
                    'type': model_type,
                    'langs': model_langs,
                    'path': os.path.join(models_dir, filename)
                })
        
        # We need a matching pair of phoneme and word models with the SAME language set
        # Sort candidates by number of languages (prefer smaller models, closer to what we want)
        candidates.sort(key=lambda x: len(x['langs']))
        
        # Group by language set
        from collections import defaultdict
        models_by_langs = defaultdict(dict)
        
        for c in candidates:
            # Create a frozenset key for the languages to group them
            lang_key = frozenset(c['langs'])
            models_by_langs[lang_key][c['type']] = c['path']
            
        # Find the best language set that has both models
        # Since we sorted candidates by length, we can iterate through the sorted unique keys
        # But dictionary order isn't guaranteed to match sort order of items inserted.
        # Let's just look for the best match.
        
        best_match = None
        min_len = float('inf')
        
        for lang_key, models in models_by_langs.items():
            if 'phonemes' in models and 'words' in models:
                if len(lang_key) < min_len:
                    min_len = len(lang_key)
                    best_match = (models['phonemes'], models['words'])
        
        if best_match:
            logger.info(f"Found models covering {lang1} and {lang2}:")
            logger.info(f"  Phoneme: {best_match[0]}")
            logger.info(f"  Word: {best_match[1]}")
            return best_match
                
        logger.warning(f"Could not find matching phoneme and word models for {lang1} and {lang2} in {models_dir}")
        return None

    def compare_words(self, word1: str, lang1: str, word2: str, lang2: str, data_dir: str) -> Dict:
        """
        Compare two words from potentially different languages.
        
        Args:
            word1: First word
            lang1: Language of first word
            word2: Second word
            lang2: Language of second word
            data_dir: Data directory path
            
        Returns:
            Dictionary with comparison results
        """
        # Load phoneme dictionaries
        dict1 = self.load_phoneme_dictionary(lang1, data_dir)
        dict2 = self.load_phoneme_dictionary(lang2, data_dir)
        
        # Get phonemes
        phonemes1 = dict1.get(word1)
        phonemes2 = dict2.get(word2)
        
        if phonemes1 is None:
            logger.error(f"Word '{word1}' not found in {lang1} dictionary")
            return None
        
        if phonemes2 is None:
            logger.error(f"Word '{word2}' not found in {lang2} dictionary")
            return None
        
        # Compare
        results = self.compare_phoneme_sequences(word1, phonemes1, word2, phonemes2)
        
        return {
            'word1': word1,
            'lang1': lang1,
            'phonemes1': phonemes1,
            'word2': word2,
            'lang2': lang2,
            'phonemes2': phonemes2,
            **results
        }


def run_word_comparison(word1: str, lang1: str, word2: str, lang2: str, 
         phoneme_model_path: str = None, word_model_path: str = None, 
         data_dir: str = 'data', device: str = 'cpu'):
    """
    Compare words across languages using trained CBOW embeddings.
    
    Args:
        word1: First word
        lang1: Language of first word (e.g., pl, en)
        word2: Second word
        lang2: Language of second word (e.g., pl, en)
        phoneme_model_path: Path to trained phoneme model (optional, auto-discovered if None)
        word_model_path: Path to trained word model (optional, auto-discovered if None)
        data_dir: Data directory (default: 'data')
        device: Device to use ('cpu' or 'cuda', default: 'cpu')
    """
    # Resolve paths to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Use provided data_dir or default to project_root/data
    if not os.path.isabs(data_dir):
        # If relative path provided, resolve from project root
        data_dir = os.path.join(project_root, data_dir.lstrip('./'))
    
    
    def discover_models(l1, l2, d_dir):
        # This duplicates logic from WordComparator.find_models_for_languages
        # Ideally we should refactor this to be a static method or standalone function
        if d_dir.endswith('data'):
            models_dir = os.path.join(os.path.dirname(d_dir), 'models')
        else:
            models_dir = os.path.join(d_dir, '../models')
        models_dir = os.path.normpath(models_dir)
        
        if not os.path.exists(models_dir):
            return None, None
            
        required = {l1, l2}
        candidates = []
        
        for fname in os.listdir(models_dir):
            if not fname.endswith('.pt'): continue
            
            if fname.startswith('cbow_phonemes_'):
                m_type = 'phonemes'
                prefix_len = 14
            elif fname.startswith('cbow_words_'):
                m_type = 'words'
                prefix_len = 11
            else:
                continue
                
            langs = set(fname[prefix_len:-3].split('_'))
            if required.issubset(langs):
                candidates.append({'type': m_type, 'langs': langs, 'path': os.path.join(models_dir, fname)})
        
        from collections import defaultdict
        by_langs = defaultdict(dict)
        for c in candidates:
            by_langs[frozenset(c['langs'])][c['type']] = c['path']
            
        best = None
        min_len = float('inf')
        for k, v in by_langs.items():
            if 'phonemes' in v and 'words' in v:
                if len(k) < min_len:
                    min_len = len(k)
                    best = (v['phonemes'], v['words'])
        return best

    if phoneme_model_path is None or word_model_path is None:
        logger.info(f"Models not specified. Attempting auto-discovery for {lang1}-{lang2}...")
        found_phoneme, found_word = discover_models(lang1, lang2, data_dir)
        
        if found_phoneme and found_word:
            phoneme_model_path = found_phoneme
            word_model_path = found_word
            logger.info(f"Auto-discovered models:\n  Phoneme: {phoneme_model_path}\n  Word: {word_model_path}")
        else:
            logger.error("Could not auto-discover models. Please specify paths manually.")
            return

    # Initialize comparator
    comparator = WordComparator(phoneme_model_path, word_model_path, device=device)
    
    # Compare words
    logger.info(f"Comparing '{word1}' ({lang1}) with '{word2}' ({lang2})")
    results = comparator.compare_words(word1, lang1, word2, lang2, data_dir)
    
    if results:
        print("\n" + "=" * 60)
        print("WORD COMPARISON RESULTS")
        print("=" * 60)
        print(f"Word 1: {results['word1']} ({results['lang1']})")
        print(f"Phonemes: {' '.join(results['phonemes1'])}")
        print(f"\nWord 2: {results['word2']} ({results['lang2']})")
        print(f"Phonemes: {' '.join(results['phonemes2'])}")
        print(f"\nSimilarity Metrics:")
        print(f"  Cosine Similarity: {results['cosine_similarity']:.4f}")
        print(f"  Euclidean Distance: {results['euclidean_distance']:.4f}")
        print("=" * 60)
    else:
        print("Comparison failed. Check logs for details.")


if __name__ == "__main__":
    # Example usage with auto-discoveryS
    run_word_comparison(word1='kot', lang1='pl', word2='cat', lang2='en')
