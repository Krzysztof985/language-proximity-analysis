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
from src.embedding_service.data.data_pipeline import run_data_pipeline

# Set up logger
logger = setup_logger(__name__, "word_comparison.log")


class WordComparator:
    """
    Compare words across languages using CBOW embeddings.
    """
    
    
    def __init__(self, phoneme_model_path: str | None = None, word_model_path: str | None = None, 
                 languages: List[str] | None = None, data_dir: str = 'data', device: str = 'auto'):
        """
        Initialize the comparator with trained phoneme and word models.
        
        Args:
            phoneme_model_path: Path to the trained phoneme model (.pt file), default: None - then auto-discover
            word_model_path: Path to the trained word model (.pt file) default: None - then auto-discover
            languages: List of languages to support (required if models are not provided)
            data_dir: Data directory path (default: 'data')
            device: Device to run on ('cpu' or 'cuda' or 'auto')
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:   
            self.device = device
            
        # Resolve data_dir to absolute path if needed
        if not os.path.isabs(data_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            data_dir = os.path.join(project_root, data_dir.lstrip('./'))
        
        if phoneme_model_path is None or word_model_path is None:
            if languages is None:
                raise ValueError("If model paths are not provided, 'languages' list must be specified.")
            
            logger.info(f"[INFO] Model paths not provided. Auto-discovering for languages: {languages}")
            
            # Ensure data availability using data pipeline
            logger.info("[INFO] Ensuring phoneme data is available...")
            run_data_pipeline(languages)
            
            # Auto-discover models
            found_models = self.find_models_for_languages(languages, data_dir)
            
            if found_models:
                phoneme_model_path, word_model_path = found_models
                logger.info(f"[INFO] Auto-discovered models:\n  Phoneme: {phoneme_model_path}\n  Word: {word_model_path}")
            else:
                raise FileNotFoundError(f"Could not find suitable models for languages: {languages}")

        # Load phoneme model
        
        # Load phoneme model
        logger.info(f"[INFO] Loading phoneme model from {phoneme_model_path}")
        phoneme_checkpoint = torch.load(phoneme_model_path, map_location=self.device)
        
        self.phoneme_vocab_size = phoneme_checkpoint['vocab_size']
        self.phoneme_embedding_dim = phoneme_checkpoint['embedding_dim']
        self.phoneme_to_idx = phoneme_checkpoint['word_to_idx']
        self.idx_to_phoneme = phoneme_checkpoint['idx_to_word']
        
        self.phoneme_model = CBOWModel(self.phoneme_vocab_size, self.phoneme_embedding_dim)
        self.phoneme_model.load_state_dict(phoneme_checkpoint['model_state_dict'])
        self.phoneme_model.to(self.device)
        self.phoneme_model.eval()
        
        logger.info(f"[INFO] Phoneme model loaded. Vocab size: {self.phoneme_vocab_size}, Embedding dim: {self.phoneme_embedding_dim}")
        
        # Load word model
        logger.info(f"[INFO] Loading word model from {word_model_path}")
        word_checkpoint = torch.load(word_model_path, map_location=self.device)
        
        self.word_vocab_size = word_checkpoint['vocab_size']
        self.word_embedding_dim = word_checkpoint['embedding_dim']
        self.char_to_idx = word_checkpoint['word_to_idx']
        self.idx_to_char = word_checkpoint['idx_to_word']
        
        self.word_model = CBOWModel(self.word_vocab_size, self.word_embedding_dim)
        self.word_model.load_state_dict(word_checkpoint['model_state_dict'])
        self.word_model.to(self.device)
        self.word_model.eval()
        
        logger.info(f"[INFO] Word model loaded. Vocab size: {self.word_vocab_size}, Embedding dim: {self.word_embedding_dim}")
        
        # Combined embedding dimension
        self.combined_embedding_dim = self.phoneme_embedding_dim + self.word_embedding_dim
        logger.info(f"[INFO] Combined embedding dimension: {self.combined_embedding_dim}")
    
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
                logger.warning(f"[WARNING] Phoneme '{phoneme}' not in vocabulary, using <UNK>")
            else:
                logger.warning(f"[WARNING] Phoneme '{phoneme}' not in vocabulary, skipping")
                continue
            
            # Get embedding
            idx_tensor = torch.tensor([idx], dtype=torch.long).to(self.device)
            with torch.no_grad():
                embedding = self.phoneme_model.embeddings(idx_tensor).cpu().numpy()[0]
            embeddings.append(embedding)
        
        if not embeddings:
            logger.warning("[WARNING] No valid phonemes found, returning zero vector")
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
                logger.warning(f"[WARNING] Character '{char}' not in vocabulary, using <UNK>")
            else:
                logger.warning(f"[WARNING] Character '{char}' not in vocabulary, skipping")
                continue
            
            # Get embedding
            idx_tensor = torch.tensor([idx], dtype=torch.long).to(self.device)
            with torch.no_grad():
                embedding = self.word_model.embeddings(idx_tensor).cpu().numpy()[0]
            embeddings.append(embedding)
        
        if not embeddings:
            logger.warning("[WARNING] No valid characters found, returning zero vector")
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
            logger.warning(f"[WARNING] Phoneme file not found for {lang} at {file_path}")
            return phoneme_dict
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    word = parts[0]
                    ipa_str = parts[1]
                    phonemes = ipa_str.split()
                    phoneme_dict[word] = phonemes
        
        logger.info(f"[INFO] Loaded {len(phoneme_dict)} words for {lang}")
        return phoneme_dict
    
    
    def find_models_for_languages(self, languages: List[str], data_dir: str) -> Tuple[str, str]:
        """
        Find phoneme and word models covering the given list of languages.
        
        Args:
            languages: List of language codes
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
            logger.warning(f"[WARNING] Models directory not found at {models_dir}")
            return None
            
        required_langs = set(languages)
        
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
            logger.info(f"[INFO] Found models covering {required_langs}:")
            logger.info(f"[INFO]   Phoneme: {best_match[0]}")
            logger.info(f"[INFO]   Word: {best_match[1]}")
            return best_match
                
        logger.warning(f"[WARNING] Could not find matching phoneme and word models covering {required_langs} in {models_dir}")
        return None

    def compare_words(self, word1: str, lang1: str, word2: str, lang2: str, data_dir: str = 'data') -> Dict:
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
            logger.error(f"[ERROR] Word '{word1}' not found in {lang1} dictionary")
            return None
        
        if phonemes2 is None:
            logger.error(f"[ERROR] Word '{word2}' not found in {lang2} dictionary")
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

