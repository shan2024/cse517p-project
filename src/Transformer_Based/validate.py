import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json
import torch.nn.functional as F
from character_transformer_model import CharacterTransformer
from utils import build_vocab
import os


context_length = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load validation data from CSVs
dev1 = pd.read_csv("data/parsed_data/dev_nasa.csv")
dev2 = pd.read_csv("data/parsed_data/dev_trek.csv")
combined = pd.concat([dev1['dialogue'], dev2['dialogue']], ignore_index=True)
text = "\n".join(combined.dropna().astype(str).tolist())

# Load vocab which was saved before
vocab_path = os.path.join("src", "Transformer_Based", "char_to_index.json")
with open(vocab_path, "r", encoding="utf-8") as f:
    char_to_index = json.load(f)
index_to_char = {i: ch for ch, i in char_to_index.items()}
vocab_size = len(char_to_index)

# Rebuild validation dataset
class ValidationCharDataset(Dataset):
    def __init__(self, text, vocab, context_len=32):
        self.context_len = context_len
        self.vocab = vocab
        self.data = [vocab[c] for c in text if c in vocab]

    def __len__(self):
        return len(self.data) - self.context_len

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.context_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+self.context_len], dtype=torch.long)
        return x, y

val_dataset = ValidationCharDataset(text, char_to_index, context_length)
val_loader = DataLoader(val_dataset, batch_size=1)

# Load model
model = CharacterTransformer(vocab_size).to(device)
model.load_state_dict(torch.load("character_transformer.pt", map_location=device))
model.eval()

# Evaluate
top1_correct = 0
top3_correct = 0
total = 0

with torch.no_grad():
    for inputs, target in val_loader:
        inputs, target = inputs.to(device), target.to(device)
        output = model(inputs)
        probs = F.softmax(output[0], dim=0)
        top3 = torch.topk(probs, k=3).indices

        total += 1
        if target.item() == top3[0].item():
            top1_correct += 1
        if target.item() in top3:
            top3_correct += 1

# Report results
print(f"\nValidation samples: {total}")
print(f"Top-1 Accuracy: {100 * top1_correct / total:.2f}%")
print(f"Top-3 Accuracy: {100 * top3_correct / total:.2f}%")
