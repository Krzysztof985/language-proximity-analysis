#!/usr/bin/env python3
"""
Visualize CBOW training progress.
"""
import argparse
import json
import matplotlib.pyplot as plt
import os


def plot_training_loss(history_path, output_path=None, show=True):
    """
    Plot training loss curve from history file.
    
    Args:
        history_path: Path to training history JSON file
        output_path: Path to save plot (optional)
        show: Whether to display the plot
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    losses = history['losses']
    config = history['config']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot loss
    epochs = range(1, len(losses) + 1)
    plt.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Labels and title
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    # Build title from config
    langs = '_'.join(config.get('languages', ['unknown']))
    data_type = config.get('data_type', 'words')
    title = f"CBOW Training Loss\n{data_type.capitalize()} ({langs})"
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add info text
    info_text = (
        f"Embedding dim: {config.get('embedding_dim', 'N/A')}\n"
        f"Window size: {config.get('window_size', 'N/A')}\n"
        f"Batch size: {config.get('batch_size', 'N/A')}\n"
        f"Learning rate: {config.get('learning_rate', 'N/A')}\n"
        f"Vocab size: {history.get('vocab_size', 'N/A')}"
    )
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Final loss annotation
    final_loss = losses[-1]
    plt.annotate(f'Final: {final_loss:.4f}',
                xy=(len(losses), final_loss),
                xytext=(10, 0), textcoords='offset points',
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.legend(loc='upper right')
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


def main():
    parser = argparse.ArgumentParser(description='Visualize CBOW training progress')
    parser.add_argument('history_file', type=str,
                       help='Path to training history JSON file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Path to save plot (optional)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display the plot')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.history_file):
        print(f"Error: History file not found: {args.history_file}")
        return
    
    # Auto-generate output path if not provided
    if args.output is None and args.no_show:
        base_name = os.path.splitext(args.history_file)[0]
        args.output = f"{base_name}_loss.png"
    
    plot_training_loss(args.history_file, args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
