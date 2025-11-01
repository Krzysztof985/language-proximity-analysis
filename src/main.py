import os
import networkx as nx
import matplotlib.pyplot as plt

from src.utils.overall_similarity import add_connection
from utils.file_utils import get_words_from_file, save_words_to_file, save_similarity_matrix
from utils.translate import translate_words
from utils.similarity import compute_similarity
from utils.overall_similarity import diagonal_average

# Config
languages = ["en", "pl", "es", "fr", "de", "pt", "it", "sl", "sk", "sv"]
data_dir = "../data"
results_dir = "../results"

os.makedirs(f"{results_dir}/translations", exist_ok=True)
os.makedirs(f"{results_dir}/similarities", exist_ok=True)

for filename in os.listdir(data_dir):
    if not filename.endswith(".txt"):
        continue

    topic = filename.replace(".txt", "")
    print(f"\n=== Processing topic: {topic} ===")
    G = nx.Graph() # Graph will be stored here
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
            # Graph creating
            outcome = diagonal_average(matrix) * 100
            add_connection(G, lang1, lang2, f"{round(outcome, 2)}%")
            pos = nx.spiral_layout(G)

            nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=700)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"),font_size=5, label_pos=0.6)

            plt.title(f"{topic} related words similarity")
            # Graph saving
            plt.savefig(f"{results_dir}/{topic}_similarity_graph.png", format="png", dpi=300, bbox_inches="tight")
            plt.close()


print("\n✅ All topics processed successfully!\nCheck results folder")
