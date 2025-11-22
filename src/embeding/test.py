import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import string
import random


# ============================================
# 1. Tokenizer znakowy
# ============================================
class CharTokenizer:
    def __init__(self):
        # zbiór znaków — możesz dodać swoje
        self.chars = ["<pad>", "<unk>"] + list(string.ascii_lowercase) + list("ąćęłńóśźż")
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, word, max_len=20):
        word = word.lower()
        ids = []
        for ch in word:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            else:
                ids.append(self.stoi["<unk>"])

        # padding do max_len
        if len(ids) < max_len:
            ids += [self.stoi["<pad>"]] * (max_len - len(ids))
        return torch.tensor(ids[:max_len])

    def vocab_size(self):
        return len(self.chars)


# ============================================
# 2. Dataset — lista słów
# ============================================
class WordDataset(Dataset):
    def __init__(self, words, tokenizer, max_len=20):
        self.words = words
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        x = self.tokenizer.encode(word, self.max_len)
        # target to samo słowo (autoencoder)
        y = x.clone()
        return x, y


# ============================================
# 3. Model — Char-CNN autoencoder
# ============================================
class CharEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, out_dim=256):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, embed_dim)

        # Convolutions to capture local character patterns
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(256, out_dim)

    def forward(self, x):
        # x: (batch, seq)
        x = self.char_embed(x)                  # (batch, seq, embed_dim)
        x = x.transpose(1, 2)                   # (batch, embed_dim, seq)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        x = self.pool(x).squeeze(-1)            # (batch, 256)
        z = self.linear(x)                      # (batch, out_dim)
        return z


class CharDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, seq_len=20):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(hidden_dim, seq_len * embed_dim)
        self.deconv_embed = nn.Linear(embed_dim, vocab_size)

    def forward(self, z):
        # z: (batch, hidden_dim)
        x = self.fc(z)                          # (batch, seq * embed_dim)
        x = x.view(z.size(0), self.seq_len, -1) # (batch, seq, embed_dim)
        logits = self.deconv_embed(x)           # (batch, seq, vocab)
        return logits


class CharAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, seq_len=20):
        super().__init__()
        self.encoder = CharEncoder(vocab_size, embed_dim, hidden_dim)
        self.decoder = CharDecoder(vocab_size, embed_dim, hidden_dim, seq_len)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.decoder(z)
        return logits, z


# ============================================
# 4. Trening modelu
# ============================================
def train_model(words):
    tokenizer = CharTokenizer()
    dataset = WordDataset(words, tokenizer)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = CharAutoencoder(vocab_size=tokenizer.vocab_size())
    optim_ = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(50):
        for x, y in dataloader:
            logits, z = model(x)
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size()),
                             y.reshape(-1))
            optim_.zero_grad()
            loss.backward()
            optim_.step()

        print(f"Epoch {epoch+1}, Loss = {loss.item():.4f}")

    return model, tokenizer


# ============================================
# 5. Używanie embeddingów
# ============================================
def embed_word(word, model, tokenizer):
    x = tokenizer.encode(word).unsqueeze(0)
    with torch.no_grad():
        _, z = model(x)
    return z.squeeze(0)


def example_usage():
    words = ["kot", "koty", "kotek", "pies", "psa", "rzeka", "żeka", "żaba", "rzaba", "biegać", "biegacz" , "cat", "dog", "river", "frog", "run", "runner"]

    model, tok = train_model(words)

    v1 = embed_word("kot", model, tok)
    v2 = embed_word("dog", model, tok)
    v3 = embed_word("rzeka", model, tok)
    v4 = embed_word("cat", model, tok)


    print(v1.shape)
    
    print("Similarity(kot, cat):",
          torch.cosine_similarity(v1, v4, dim=0).item())
    print("Similarity(rzeka, dog):",
          torch.cosine_similarity(v3, v2, dim=0).item())


if __name__ == "__main__":
    example_usage()
