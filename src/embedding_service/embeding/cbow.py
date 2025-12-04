import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import os
import sys
import logging
# Add project root to sys.path for logging import
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.logger.logging_config import setup_logger

# Set up logger for this module
logger = setup_logger(__name__, 'cbow.log', level=logging.DEBUG)


def build_vocab(sequences):
    """
    Build vocabulary from sequences of tokens.
    
    Args:
        sequences: List of sequences (words as strings or phoneme lists)
        
    Returns:
        List of unique tokens (vocabulary)
    """
    # Flatten all sequences to count tokens
    all_tokens = []
    for seq in sequences:
        all_tokens.extend(list(seq))
        
    counter = Counter(all_tokens)
    # Sort by frequency for consistency
    vocab = sorted(counter.keys())
    # Add padding token and unknown token
    vocab.append('.')
    vocab.append('<UNK>')
    logger.debug(f"[DEBUG] Vocabulary: {vocab}")    
    return vocab


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
    def __init__(self, sequences, window_size=2, vocab=None):
        """
        Args:
            sequences: List of sequences (words as strings or phoneme lists)
            window_size: Number of tokens on each side of target token
            vocab: Optional vocabulary list. If provided, it will be used instead of building from sequences.
        """
        self.sequences = sequences
        self.window_size = window_size
        
        # Build vocabulary
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = self._build_vocab(sequences)
            
        self.vocab_size = len(self.vocab)
        self.word_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_word = {idx: token for token, idx in self.word_to_idx.items()}
        
        # Generate training pairs
        self.data = self._generate_training_data()
        
    def _build_vocab(self, sequences):
        """Build vocabulary from sequences of tokens."""
        return build_vocab(sequences)
    
    def _generate_training_data(self):
        """Generate (context, target) pairs respecting sequence boundaries."""
        data = []
        padding_idx = self.word_to_idx['.']
        unk_idx = self.word_to_idx.get('<UNK>')
        max_context_size =  self.window_size # * 2
        
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
                        token = tokens[j]
                        if token in self.word_to_idx:
                            context.append(self.word_to_idx[token])
                        elif unk_idx is not None:
                            context.append(unk_idx)
                        # else: skip unknown words if no UNK token (shouldn't happen with current setup)
                
                # Skip if no context (shouldn't happen if len >= 2 and window >= 1)
                if not context:
                    continue
                
                # Pad context to fixed size
                while len(context) < max_context_size:
                    context.append(padding_idx)
                    
                target_token = tokens[i]
                if target_token in self.word_to_idx:
                    target = self.word_to_idx[target_token]
                elif unk_idx is not None:
                    target = unk_idx
                else:
                    continue # Skip if target is unknown and no UNK token
                    
                data.append((context, target))
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def train_cbow(model, dataset, val_dataset=None, epochs=10, batch_size=64, learning_rate=0.001, patience=3, min_delta=0.001, device='cpu'):
    """
    Train the CBOW model.
    
    Args:
        model: CBOWModel instance
        dataset: CBOWDataset instance (training)
        val_dataset: CBOWDataset instance (validation), optional
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in validation loss to qualify as an improvement
        device: Device to train on ('cpu' or 'cuda')
    
    Returns:
        Trained model and list of losses
    """
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if val_dataset:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
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
        
        log_msg = f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}"
        
        # Validation loop
        if val_dataset:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for context, target in val_dataloader:
                    context = context.to(device)
                    target = target.to(device)
                    output = model(context)
                    loss = criterion(output, target)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)
            log_msg += f", Val Loss: {avg_val_loss:.4f}"
            
            # Early stopping check
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
                # logger.info(f"New best model found (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(log_msg)
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    model.load_state_dict(best_model_state)
                    break
            
        logger.info(log_msg)
    
    # Load best model if validation was performed
    if val_dataset and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, losses, val_losses


def evaluate_model(model, dataset, batch_size=64, device='cpu'):
    """
    Evaluate the model on a given dataset.
    
    Args:
        model: Trained CBOWModel
        dataset: CBOWDataset for evaluation
        batch_size: Batch size for evaluation
        device: Device to evaluate on
        
    Returns:
        Average loss on the dataset
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    
    with torch.no_grad():
        for context, target in dataloader:
            context = context.to(device)
            target = target.to(device)
            output = model(context)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def get_word_embedding(model, character, vocab, device='cpu'):
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
    if character not in vocab:
        raise ValueError(f"Character '{character}' not in vocabulary")
    
    idx = torch.tensor([vocab[character]], dtype=torch.long).to(device)
    with torch.no_grad():
        embedding = model.embeddings(idx).cpu().numpy()[0]
    
    return embedding



