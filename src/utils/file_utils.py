import os
import csv
from typing import Dict, List, Optional

def get_words_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [w.strip() for w in f.readlines() if w.strip()]

def save_words_to_file(words, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for w in words:
            f.write(w + "\n")

def load_translations_csv(csv_path: str) -> Optional[Dict[str, List[str]]]:
    """Load translations from CSV file.
    
    Returns:
        Dict with language codes as keys and word lists as values, or None if file doesn't exist
    """
    if not os.path.exists(csv_path):
        return None
        
    try:
        translations = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Language codes
            
            # Initialize lists for each language
            for lang in headers:
                translations[lang] = []
            
            # Read word rows
            for row in reader:
                for i, word in enumerate(row):
                    if i < len(headers):  # Safety check
                        translations[headers[i]].append(word.strip())
        
        return translations
    except Exception as e:
        print(f"Error loading translations from {csv_path}: {e}")
        return None

def save_translations_csv(translations: Dict[str, List[str]], csv_path: str) -> None:
    """Save translations to CSV file.
    
    Args:
        translations: Dict with language codes as keys and word lists as values
        csv_path: Path to save the CSV file
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Get language order (put 'en' first if present)
    languages = list(translations.keys())
    if 'en' in languages:
        languages.remove('en')
        languages.insert(0, 'en')
    
    # Get max word count to handle different lengths
    max_words = max(len(words) for words in translations.values()) if translations else 0
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(languages)
        
        # Write word rows
        for i in range(max_words):
            row = []
            for lang in languages:
                words = translations.get(lang, [])
                word = words[i] if i < len(words) else ''
                row.append(word)
            writer.writerow(row)

def save_similarity_matrix(words1, words2, matrix, file_path):
    """
    Zapisuje pełną macierz podobieństw do pliku CSV.
    Wiersze: words1, Kolumny: words2
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + words2)  # nagłówki kolumn
        for w1, row in zip(words1, matrix):
            writer.writerow([w1] + [f"{v:.2f}" for v in row])
