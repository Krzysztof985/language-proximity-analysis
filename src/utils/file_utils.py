import os
import csv

def get_words_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [w.strip() for w in f.readlines() if w.strip()]

def save_words_to_file(words, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for w in words:
            f.write(w + "\n")

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
