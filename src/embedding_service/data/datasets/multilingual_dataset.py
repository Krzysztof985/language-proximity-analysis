import os
import torch
from torch.utils.data import Dataset

class MultilingualWordDataset(Dataset):
    """
    Dataset for loading character-level data from multilingual word lists.
    Returns: (character, language_label)
    """
    def __init__(self, languages, data_dir):
        self.languages = languages
        self.data_dir = data_dir
        self.samples = []
        self.lang_to_idx = {lang: idx for idx, lang in enumerate(languages)}

        self.samples = list(self._load_data(languages, data_dir))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @staticmethod
    def _load_data(languages, data_dir):
        lang_to_idx = {lang: idx for idx, lang in enumerate(languages)}
        for lang in languages:
            file_path = os.path.join(data_dir, lang, "phonemes.txt")
            if not os.path.exists(file_path):
                print(f"Warning: Word file not found for {lang} at {file_path}")
                continue
            
            lang_idx = lang_to_idx[lang]
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 1:
                        word = parts[0]
                        if word:
                            yield (word, lang_idx)

class MultilingualPhonemeDataset(Dataset):
    """
    Dataset for loading phoneme-level data from multilingual IPA dictionaries.
    Returns: (phoneme, language_label)
    """
    def __init__(self, languages, data_dir):
        self.languages = languages
        self.data_dir = data_dir
        self.samples = []
        self.lang_to_idx = {lang: idx for idx, lang in enumerate(languages)}

        self.samples = list(self._load_data(languages, data_dir))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @staticmethod
    def _load_data(languages, data_dir):
        lang_to_idx = {lang: idx for idx, lang in enumerate(languages)}
        for lang in languages:
            file_path = os.path.join(data_dir, lang, "phonemes.txt")
            if not os.path.exists(file_path):
                print(f"Warning: Phoneme file not found for {lang} at {file_path}")
                continue
            
            lang_idx = lang_to_idx[lang]
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        ipa_str = parts[1]
                        # WikiPron IPA is often space-separated (e.g., "a t͡s ɛ")
                        phonemes = ipa_str.split()
                        if phonemes:
                            yield (phonemes, lang_idx)