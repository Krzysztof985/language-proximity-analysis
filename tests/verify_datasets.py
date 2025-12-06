import sys
import os
import torch

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.embedding_service.data.datasets.multilingual_dataset import MultilingualWordDataset, MultilingualPhonemeDataset

def verify_datasets():
    print("="*50)
    print("Verifying Datasets")
    print("="*50)

    base_data_dir = os.path.join(project_root, "data")
    languages = ["pl"] # Testing with Polish for now

    # --- Verify Word Dataset ---
    print("\n--- Verifying MultilingualWordDataset ---")
    word_dataset = MultilingualWordDataset(languages, base_data_dir)
    print(f"Loaded Word Dataset with {len(word_dataset)} characters.")

    # Reconstruct words from dataset (approximation, as dataset flattens)
    # We can't easily reconstruct exact words because we lost boundaries.
    # But we can check if the sequence of characters matches the file content.
    
    dataset_chars = [item[0] for item in word_dataset]
    dataset_content = "".join(dataset_chars)
    
    # Read original file to compare
    # Note: The dataset implementation currently reads phonemes.txt
    original_content = ""
    phonemes_path = os.path.join(base_data_dir, "pl", "phonemes.txt")
    
    print(f"Reading original file: {phonemes_path}")
    if os.path.exists(phonemes_path):
        with open(phonemes_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Logic from MultilingualWordDataset (current implementation)
                # word = line.strip()
                # for char in word: ...
                # So we expect the dataset to contain all characters from line.strip()
                original_content += line.strip()
    
    # Write dataset content to file for inspection
    output_words_path = os.path.join(os.path.dirname(__file__), "output_words_dataset.txt")
    with open(output_words_path, 'w', encoding='utf-8') as f:
        f.write(dataset_content)
    print(f"Wrote dataset content to: {output_words_path}")

    # Compare
    if dataset_content == original_content:
        print("[OK] Word Dataset content matches file content (based on current implementation logic).")
    else:
        print("[ERROR] Word Dataset content DOES NOT match file content.")
        print(f"Length Dataset: {len(dataset_content)}")
        print(f"Length Original: {len(original_content)}")
        # Print first diff
        for i in range(min(len(dataset_content), len(original_content))):
            if dataset_content[i] != original_content[i]:
                print(f"First mismatch at index {i}: Dataset='{dataset_content[i]}', Original='{original_content[i]}'")
                print(f"Context Dataset: {dataset_content[i:i+20]}")
                print(f"Context Original: {original_content[i:i+20]}")
                break

    # Check if we are accidentally reading IPA chars as word chars
    # If the dataset reads the whole line "word\tipa", it includes tabs and IPA symbols.
    if '\t' in dataset_content:
        print("\n[WARNING] Tab character found in Word Dataset content!")
        print("This suggests the dataset is reading the entire line (Word + IPA) instead of just the Word.")
    
    # --- Verify Phoneme Dataset ---
    print("\n--- Verifying MultilingualPhonemeDataset ---")
    phoneme_dataset = MultilingualPhonemeDataset(languages, base_data_dir)
    print(f"Loaded Phoneme Dataset with {len(phoneme_dataset)} phonemes.")

    dataset_phonemes = [item[0] for item in phoneme_dataset]
    
    # Read original file to compare
    original_phonemes = []
    if os.path.exists(phonemes_path):
        with open(phonemes_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    ipa_str = parts[1]
                    phonemes = ipa_str.split()
                    original_phonemes.extend(phonemes)

    # Write dataset content to file
    output_phonemes_path = os.path.join(os.path.dirname(__file__), "output_phonemes_dataset.txt")
    with open(output_phonemes_path, 'w', encoding='utf-8') as f:
        f.write(" ".join(dataset_phonemes))
    print(f"Wrote dataset content to: {output_phonemes_path}")

    if dataset_phonemes == original_phonemes:
        print("[OK] Phoneme Dataset content matches file content.")
    else:
        print("[ERROR] Phoneme Dataset content DOES NOT match file content.")
        print(f"Count Dataset: {len(dataset_phonemes)}")
        print(f"Count Original: {len(original_phonemes)}")
        for i in range(min(len(dataset_phonemes), len(original_phonemes))):
            if dataset_phonemes[i] != original_phonemes[i]:
                print(f"First mismatch at index {i}: Dataset='{dataset_phonemes[i]}', Original='{original_phonemes[i]}'")
                break

if __name__ == "__main__":
    verify_datasets()
