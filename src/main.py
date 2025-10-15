import os
from utils.file_utils import get_words_from_file, save_words_to_file, save_similarity_matrix
from utils.translate import translate_words
from utils.similarity import compute_similarity

# Config
languages = ["en", "pl", "es"]
data_dir = "data"
results_dir = "results"

os.makedirs(f"{results_dir}/translations", exist_ok=True)
os.makedirs(f"{results_dir}/similarities", exist_ok=True)

for filename in os.listdir(data_dir):
    if not filename.endswith(".txt"):
        continue

    topic = filename.replace(".txt", "")
    print(f"\n=== Processing topic: {topic} ===")

    words = get_words_from_file(os.path.join(data_dir, filename))
    print(f"Loaded {len(words)} words from {filename}")

    translations = {lang: translate_words(words, lang) for lang in languages}

    for lang, trans_words in translations.items():
        save_words_to_file(trans_words, f"{results_dir}/translations/{topic}_{lang}.txt")

    for i in range(len(languages)):
        for j in range(i + 1, len(languages)):
            lang1, lang2 = languages[i], languages[j]
            matrix = [[compute_similarity(w1, w2) for w2 in translations[lang2]] for w1 in translations[lang1]]
            save_similarity_matrix(translations[lang1], translations[lang2], matrix,
                                   f"{results_dir}/similarities/{topic}_{lang1}_{lang2}.csv")

print("\n✅ All topics processed successfully!\nCheck results folder")