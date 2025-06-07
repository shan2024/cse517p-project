import torch
from torch.utils.data import Dataset
import pandas as pd

from data_parsing.helpers import DatasetFileLoader

class CharDataset(Dataset):
    def __init__(self, data: str,context_len=128):
        self.context_len = context_len
        self.data_indices = data 

    def __len__(self):
        return len(self.data_indices) - self.context_len

    def __getitem__(self, idx):
        input_seq = self.data_indices[idx:idx+self.context_len]
        target_char = self.data_indices[idx+self.context_len]
        return input_seq, target_char
    
class CharDatasetWrapper():
    # Load training data
    def __init__(self, vocab, data_directory, context_length: int = 128, dataset_fraction: float = 1):
        data_loader = DatasetFileLoader()
        data_loader.load(data_directory, dataset_fraction)

        # TODO: Fix this so getting [0] and slicing first line isn't needed
        train_data = data_loader.train_data[0][1:]

        dev_data = data_loader.dev_data[0][1:]

        train_text = "\n".join(train_data)
        dev_text = "\n".join(dev_data)

        train_data = torch.tensor([vocab[char] for char in train_text if char in vocab])
        dev_data = torch.tensor([vocab[char] for char in dev_text if char in vocab])

        self._train_dataset = CharDataset(train_data,  context_length)
        self._dev_dataset = CharDataset(dev_data, context_length)

    def train_dataset(self):
        return self._train_dataset
    
    def dev_dataset(self):
        return self._dev_dataset
