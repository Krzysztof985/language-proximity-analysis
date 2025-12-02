import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import os
import sys

# Add project root to sys.path for logging import
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.logging.logging_config import setup_logger

# Set up logger for this module
logger = setup_logger(__name__, 'cbow.log')


class CBOWModel(nn.Module):
    """
    Continuous Bag of Words (CBOW) model for Word2Vec.
    Predicts a target word from its context words.
    """
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.activation_function1 = nn.ReLU()
        self.linear2 = nn.Linear(128, vocab_size)
        # self.activation_function2 = nn.LogSoftmax(dim=-1)
        
    def forward(self, context):
        embeds = torch.sum(self.embeddings(context), dim=1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        # out = self.activation_function2(out)
        return out


class CBOWDataset(Dataset):
    """
    Dataset for CBOW training.
    Generates (context_words, target_word) pairs from a sequence of tokens.
    """
    def __init__(self, sequences, window_size=2):
        """
        Args:
            sequences: List of sequences (words as strings or phoneme lists)
            window_size: Number of tokens on each side of target token
        """
        self.sequences = sequences
        self.window_size = window_size
        
        # Build vocabulary
        self.vocab = self._build_vocab(sequences)
        self.vocab_size = len(self.vocab)
        self.word_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_word = {idx: token for token, idx in self.word_to_idx.items()}
        
        # Generate training pairs
        self.data = self._generate_training_data()
        
    def _build_vocab(self, sequences):
        """Build vocabulary from sequences of tokens."""
        # Flatten all sequences to count tokens
        all_tokens = []
        for seq in sequences:
            all_tokens.extend(list(seq))
            
        counter = Counter(all_tokens)
        # Sort by frequency for consistency
        vocab = sorted(counter.keys())
        # Add padding token
        vocab.append('.')
        return vocab
    
    def _generate_training_data(self):
        """Generate (context, target) pairs respecting sequence boundaries."""
        data = []
        padding_idx = self.word_to_idx['.']
        max_context_size = 2 * self.window_size
        
        for seq in self.sequences:
            tokens = list(seq)
            if len(tokens) < 2:
                continue
                
            for i in range(len(tokens)):
                # Get context window constrained to sequence boundaries
                start_idx = max(0, i - self.window_size)
                end_idx = min(len(tokens), i + self.window_size + 1)
                
                context = []
                for j in range(start_idx, end_idx):
                    if j != i:
                        context.append(self.word_to_idx[tokens[j]])
                
                # Skip if no context (shouldn't happen if len >= 2 and window >= 1)
                if not context:
                    continue
                
                # Pad context to fixed size
                while len(context) < max_context_size:
                    context.append(padding_idx)
                    
                target = self.word_to_idx[tokens[i]]
                data.append((context, target))
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def train_cbow(model, dataset, epochs=10, batch_size=64, learning_rate=0.001, device='cpu'):
    """
    Train the CBOW model.
    
    Args:
        model: CBOWModel instance
        dataset: CBOWDataset instance
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
    
    Returns:
        Trained model and list of losses
    """
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            context = context.to(device)
            target = target.to(device)
            
            # Forward pass
            output = model(context)
            loss = criterion(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, losses


def get_word_embedding(model, word, word_to_idx, device='cpu'):
    """
    Get the embedding vector for a word.
    
    Args:
        model: Trained CBOWModel
        word: Word to get embedding for
        word_to_idx: Dictionary mapping words to indices
        device: Device model is on
    
    Returns:
        Embedding vector as numpy array
    """
    if word not in word_to_idx:
        raise ValueError(f"Word '{word}' not in vocabulary")
    
    idx = torch.tensor([word_to_idx[word]], dtype=torch.long).to(device)
    with torch.no_grad():
        embedding = model.embeddings(idx).cpu().numpy()[0]
    
    return embedding


def find_similar_words(model, word, word_to_idx, idx_to_word, top_k=5, device='cpu'):
    """
    Find the most similar words to a given word based on cosine similarity.
    
    Args:
        model: Trained CBOWModel
        word: Word to find similar words for
        word_to_idx: Dictionary mapping words to indices
        idx_to_word: Dictionary mapping indices to words
        top_k: Number of similar words to return
        device: Device model is on
    
    Returns:
        List of (word, similarity) tuples
    """
    target_embedding = get_word_embedding(model, word, word_to_idx, device)
    
    # Get all embeddings
    all_embeddings = model.embeddings.weight.detach().cpu().numpy()
    
    # Compute cosine similarities
    similarities = np.dot(all_embeddings, target_embedding) / (
        np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(target_embedding)
    )
    
    # Get top k similar words (excluding the word itself)
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    similar_words = [(idx_to_word[idx], similarities[idx]) for idx in top_indices]
    
    return similar_words
