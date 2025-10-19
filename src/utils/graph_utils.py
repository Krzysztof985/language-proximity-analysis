import csv
import os

def save_similarity_matrix_csv(matrix, languages, topic, folder="results/graphs"):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{topic}_matrix.csv")
    with open(file_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + languages)
        for lang, row in zip(languages, matrix):
            writer.writerow([lang] + row)
