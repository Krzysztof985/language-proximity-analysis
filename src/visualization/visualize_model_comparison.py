
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Resolve paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    results_file = os.path.join(project_root, 'results', 'model_comparison_results.json')
    output_plot = os.path.join(project_root, 'results', 'similarity_vs_levenshtein.png')
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        print("Please run src/embedding_service/test_models.py first.")
        return

    # Load data
    with open(results_file, 'r') as f:
        data = json.load(f)
        
    if not data:
        print("No data found in results file.")
        return

    # Extract metrics
    words_labels = []
    cos_sims = []
    lev_sims = []
    
    for item in data:
        label = f"{item['word1']} ({item['lang1']})\nvs\n{item['word2']} ({item['lang2']})"
        words_labels.append(label)
        cos_sims.append(item['cosine_similarity'])
        lev_sims.append(item['levenshtein_similarity'])

    # Create Scatter Plot
    plt.figure(figsize=(12, 8))
    
    # Plot points
    plt.scatter(lev_sims, cos_sims, color='blue', alpha=0.7, s=100)
    
    # Add diagonal line (y=x) for reference
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Correlation')
    
    # Annotate points
    for i, label in enumerate(words_labels):
        plt.annotate(label, (lev_sims[i], cos_sims[i]), 
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, alpha=0.8)

    plt.title('Embedding Similarity vs. Levenshtein Similarity', fontsize=16)
    plt.xlabel('Levenshtein Similarity (String Edit Distance)', fontsize=12)
    plt.ylabel('Cosine Similarity (Embedding Space)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 1.05)
    plt.ylim(-0.2, 1.05) # Cosine similarity can be negative, but usually > 0 for related words
    
    # Add quadrant explanations
    plt.text(0.1, 0.9, 'High Embedding Sim\nLow String Sim\n(Hidden Relations/Cognates)', 
             fontsize=9, bbox=dict(facecolor='green', alpha=0.1))
    plt.text(0.8, 0.1, 'Low Embedding Sim\nHigh String Sim\n(False Friends?)', 
             fontsize=9, bbox=dict(facecolor='red', alpha=0.1))

    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"Plot saved to {output_plot}")
    plt.close()

if __name__ == "__main__":
    main()
