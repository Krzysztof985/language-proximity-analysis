
import os
import sys

# Add project root to path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
# Root is one level up from tests/
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.embedding_service.compare_words import WordComparator

def main():
    # Define test cases: (word1, lang1, word2, lang2)
    test_cases = [
        # Romance languages - similar
        ('gato', 'es', 'gatto', 'it'),      # Cat
        ('amigo', 'es', 'amico', 'it'),     # Friend
        
        # Slavic languages - similar
        ('kot', 'pl', 'кот', 'ru'),         # Cat
        ('dom', 'pl', 'дом', 'ru'),         # House
        ('śnieg', 'pl', 'снег', 'ru'),      # Snow
        
        # Germanic - similar
        ('cat', 'en', 'kat', 'nl'),         # Cat
        ('milk', 'en', 'melk', 'nl'),       # Milk
        
        # Distant pairs
        ('kot', 'pl', 'cat', 'en'),         # Cat (Polish vs English)
        ('pies', 'pl', 'dog', 'en'),        # Dog (Polish vs English) - different roots
        ('oko', 'pl', 'eye', 'en'),         # Eye - Cognates but look different
        
        # Finno-Ugric vs Indo-European (Very distant)
        ('kissa', 'fi', 'cat', 'en'),       # Cat
        ('vesi', 'fi', 'water', 'en'),      # Water
    ]

    print("Initializing WordComparator...")
    
    models_dir = os.path.join(project_root, 'models')
    phoneme_model = None
    word_model = None
    
    # scan for the large multilingual model we saw earlier
    for f in os.listdir(models_dir):
        if f.startswith('cbow_phonemes_') and f.endswith('.pt') and 'pl' in f and 'en' in f:
            phoneme_model = os.path.join(models_dir, f)
        if f.startswith('cbow_words_') and f.endswith('.pt') and 'pl' in f and 'en' in f:
            word_model = os.path.join(models_dir, f)
            
    if not phoneme_model or not word_model:
        print("Error: Could not find suitable multilingual models in models/ directory.")
        return

    print(f"Using Phoneme Model: {os.path.basename(phoneme_model)}")
    print(f"Using Word Model: {os.path.basename(word_model)}")
    
    comparator = WordComparator(phoneme_model, word_model)
    data_dir = os.path.join(project_root, 'data')

    import json
    from src.utils.similarity import compute_similarity

    results_data = []

    print("\nStarting Comparisons...\n")
    print(f"{'Word 1':<15} {'Lang1':<6} {'Word 2':<15} {'Lang2':<6} {'Cos Sim':<10} {'Lev Sim':<10} {'Euclidean':<10}")
    print("-" * 80)

    for w1, l1, w2, l2 in test_cases:
        try:
            # We call compare_words method
            result = comparator.compare_words(w1, l1, w2, l2, data_dir)
            
            # Calculate Levenshtein similarity
            lev_sim = compute_similarity(w1, w2)
            
            if result:
                sim = result['cosine_similarity']
                dist = result['euclidean_distance']
                print(f"{w1:<15} {l1:<6} {w2:<15} {l2:<6} {sim:<10.4f} {lev_sim:<10.4f} {dist:<10.4f}")
                
                results_data.append({
                    'word1': w1, 'lang1': l1,
                    'word2': w2, 'lang2': l2,
                    'cosine_similarity': sim,
                    'levenshtein_similarity': lev_sim,
                    'euclidean_distance': dist
                })
            else:
                 print(f"{w1:<15} {l1:<6} {w2:<15} {l2:<6} {'FAILED':<10} {lev_sim:<10.4f} {'N/A':<10}")
        except Exception as e:
             print(f"{w1:<15} {l1:<6} {w2:<15} {l2:<6} {'ERROR':<10} {'-':<10} {str(e)}")

    # Save results
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, 'model_comparison_results.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
