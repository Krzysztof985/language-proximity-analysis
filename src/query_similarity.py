#!/usr/bin/env python3
"""
Query large similarity matrices stored in HDF5 format without loading them entirely into memory.
"""
import h5py
import numpy as np
import sys
import os

def query_word_similarity(h5_file, word, lang, top_k=10):
    """
    Find most similar words for a given word without loading the entire matrix.
    
    Args:
        h5_file: Path to .h5 file
        word: Word to query
        lang: Language of the word ('pl' or 'en')
        top_k: Number of top results to return
    """
    # Open HDF5 file (memory-mapped access)
    with h5py.File(h5_file, 'r') as f:
        # Read metadata
        lang1 = f.attrs['lang1']
        lang2 = f.attrs['lang2']
        
        # Read word lists (these are small, so it's OK to load them)
        words1 = [w.decode('utf-8') if isinstance(w, bytes) else w for w in f['words1'][:]]
        words2 = [w.decode('utf-8') if isinstance(w, bytes) else w for w in f['words2'][:]]
        
        print(f"Matrix: {lang1} ({len(words1)} words) vs {lang2} ({len(words2)} words)")
        
        # Access the similarity matrix (memory-mapped, not loaded into RAM)
        similarity_matrix = f['similarity_matrix']
        
        # Find the word index
        if lang == lang1:
            if word not in words1:
                print(f"Error: Word '{word}' not found in {lang}")
                return
            
            idx = words1.index(word)
            target_words = words2
            target_lang = lang2
            
            # Load only the specific row we need (memory-efficient!)
            similarities = similarity_matrix[idx, :]
            
        elif lang == lang2:
            if word not in words2:
                print(f"Error: Word '{word}' not found in {lang}")
                return
            
            idx = words2.index(word)
            target_words = words1
            target_lang = lang1
            
            # Load only the specific column we need
            similarities = similarity_matrix[:, idx]
            
        else:
            print(f"Error: Language {lang} not in matrix (available: {lang1}, {lang2})")
            return
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        print(f"\nTop {top_k} most similar words to '{word}' ({lang}):")
        print("=" * 60)
        for rank, i in enumerate(top_indices, 1):
            similar_word = target_words[i]
            score = similarities[i]
            print(f"{rank:2d}. {similar_word:20s} ({target_lang}) - similarity: {score:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python src/query_similarity.py <h5_file> <word> <lang> [top_k]")
        print("Example: python src/query_similarity.py results/similarity_matrix_pl_en.h5 hello en 10")
        sys.exit(1)
    
    h5_file = sys.argv[1]
    
    if not os.path.exists(h5_file):
        print(f"Error: File '{h5_file}' not found")
        sys.exit(1)
    
    word = sys.argv[2]
    lang = sys.argv[3]
    top_k = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    
    query_word_similarity(h5_file, word, lang, top_k)
