import torch
from torch.utils.data import Dataset

def build_vocab(text):
    unique_chars = sorted(set(text))
    unique_chars.remove('\n')
    char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
    index_to_char = {idx: char for char, idx in char_to_index.items()}
    return char_to_index, index_to_char

class CharDataset(Dataset):
    def __init__(self, text, vocab, context_len=32):
        self.vocab = vocab
        self.context_len = context_len
        self.data_indices = [vocab[char] for char in text if char in vocab]

    def __len__(self):
        return len(self.data_indices) - self.context_len

    def __getitem__(self, idx):
        input_seq = torch.tensor(self.data_indices[idx:idx+self.context_len], dtype=torch.long)
        target_char = torch.tensor(self.data_indices[idx+self.context_len], dtype=torch.long)
        return input_seq, target_char