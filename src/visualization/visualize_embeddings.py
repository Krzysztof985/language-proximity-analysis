#!/usr/bin/env python3
"""
Visualize CBOW embeddings using dimensionality reduction.
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import sys


def load_model_and_vocab(model_path):
    """Load saved model and vocabulary."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get embeddings
    embeddings = checkpoint['model_state_dict']['embeddings.weight'].numpy()
    word_to_idx = checkpoint['word_to_idx']
    idx_to_word = checkpoint['idx_to_word']
    
    # Convert string keys to int for idx_to_word
    idx_to_word = {int(k): v for k, v in idx_to_word.items()}
    
    return embeddings, word_to_idx, idx_to_word


def visualize_embeddings_2d(embeddings, idx_to_word, method='tsne', 
                            max_words=500, output_path=None, show=True,
                            annotate_top=20):
    """
    Visualize embeddings in 2D using dimensionality reduction.
    
    Args:
        embeddings: Embedding matrix (vocab_size, embedding_dim)
        idx_to_word: Dictionary mapping indices to words
        method: 'tsne' or 'pca'
        max_words: Maximum number of words to visualize
        output_path: Path to save plot
        show: Whether to display the plot
        annotate_top: Number of words to annotate
    """
    # Limit number of words
    n_words = min(max_words, len(embeddings))
    embeddings_subset = embeddings[:n_words]
    
    # Dimensionality reduction
    if method == 'tsne':
        print(f"Running t-SNE (this may take a while for {n_words} words)...")
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, n_words-1))
        reduced = reducer.fit_transform(embeddings_subset)
        title = "t-SNE Visualization of CBOW Embeddings"
    elif method == 'pca':
        print(f"Running PCA for {n_words} words...")
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings_subset)
        var_explained = reducer.explained_variance_ratio_
        title = f"PCA Visualization of CBOW Embeddings\n(Variance explained: {var_explained[0]:.2%} + {var_explained[1]:.2%})"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Scatter plot
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], 
                        alpha=0.6, s=50, c=range(n_words), 
                        cmap='viridis', edgecolors='w', linewidth=0.5)
    
    # Annotate top words
    if annotate_top > 0:
        # Annotate first N words (usually most frequent)
        for i in range(min(annotate_top, n_words)):
            word = idx_to_word.get(i, f'idx_{i}')
            ax.annotate(word, (reduced[i, 0], reduced[i, 1]),
                       fontsize=9, alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='yellow', alpha=0.3))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Word Index (frequency order)')
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def visualize_similarity_heatmap(embeddings, idx_to_word, words,
                                 output_path=None, show=True):
    """
    Create a similarity heatmap for specified words.
    
    Args:
        embeddings: Embedding matrix
        idx_to_word: Dictionary mapping indices to words
        words: List of words to include in heatmap
        output_path: Path to save plot
        show: Whether to display the plot
    """
    from matplotlib.colors import LinearSegmentedColormap
    
    # Get word indices
    word_to_idx = {v: k for k, v in idx_to_word.items()}
    indices = []
    valid_words = []
    
    for word in words:
        if word in word_to_idx:
            indices.append(word_to_idx[word])
            valid_words.append(word)
    
    if len(indices) == 0:
        print("No valid words found in vocabulary")
        return
    
    # Get embeddings for these words
    word_embeddings = embeddings[indices]
    
    # Compute cosine similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(word_embeddings)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Custom colormap (white -> yellow -> red)
    colors = ['white', 'yellow', 'red']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('similarity', colors, N=n_bins)
    
    im = ax.imshow(similarity_matrix, cmap=cmap, vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks(range(len(valid_words)))
    ax.set_yticks(range(len(valid_words)))
    ax.set_xticklabels(valid_words, rotation=45, ha='right')
    ax.set_yticklabels(valid_words)
    
    # Add values in cells
    for i in range(len(valid_words)):
        for j in range(len(valid_words)):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Word Similarity Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to: {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize CBOW embeddings')
    parser.add_argument('model_path', type=str,
                       help='Path to saved model (.pt file)')
    parser.add_argument('--method', choices=['tsne', 'pca', 'both'], default='tsne',
                       help='Dimensionality reduction method (default: tsne)')
    parser.add_argument('--max-words', type=int, default=500,
                       help='Maximum words to visualize (default: 500)')
    parser.add_argument('--annotate', type=int, default=20,
                       help='Number of words to annotate (default: 20)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Directory to save plots')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots')
    parser.add_argument('--heatmap-words', nargs='+', default=None,
                       help='Words to include in similarity heatmap')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    embeddings, word_to_idx, idx_to_word = load_model_and_vocab(args.model_path)
    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Vocabulary size: {len(word_to_idx)}")
    
    # Determine output paths
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.model_path))[0]
    
    # Create visualizations
    methods = ['tsne', 'pca'] if args.method == 'both' else [args.method]
    
    for method in methods:
        output_path = None
        if args.output_dir:
            output_path = os.path.join(args.output_dir, f"{base_name}_{method}.png")
        
        visualize_embeddings_2d(
            embeddings, idx_to_word, method=method,
            max_words=args.max_words, output_path=output_path,
            show=not args.no_show, annotate_top=args.annotate
        )
    
    # Create heatmap if words specified
    if args.heatmap_words:
        output_path = None
        if args.output_dir:
            output_path = os.path.join(args.output_dir, f"{base_name}_heatmap.png")
        
        visualize_similarity_heatmap(
            embeddings, idx_to_word, args.heatmap_words,
            output_path=output_path, show=not args.no_show
        )


if __name__ == "__main__":
    main()
