import torch
from torch.utils.data import DataLoader
import pandas as pd
import random
import json
import os

from character_transformer_model import CharacterTransformer
from utils import build_vocab, CharDataset

parent_dir = os.path.dirname(os.path.abspath("__file__"))
data_dir = os.path.join(parent_dir, "../../data")
test_dir = os.path.join(parent_dir, "../../test")

# Load training data
nasa_df = pd.read_csv(f"{data_dir}/parsed_data/train_nasa.csv")
trek_df = pd.read_csv(f"{data_dir}/parsed_data/train_trek.csv")
combined_dialogues = pd.concat([nasa_df['dialogue'], trek_df['dialogue']], ignore_index=True)
all_lines = combined_dialogues.dropna().astype(str).tolist()

# Just used for controlling the size of the dataset used for traiing. May not use this later
sample_fraction = .01  
total_lines = len(all_lines)
sample_count = max(1, round(sample_fraction * total_lines))

random.seed(42)
sampled_lines = random.sample(all_lines, sample_count)

# Combine sampled lines into a single training string
train_text = "\n".join(sampled_lines)

print(f"Sampled {sample_count:,} lines "f"({sample_fraction:.3%}) out of {total_lines:,} total lines.")

# Build vocabulary 
char_to_index, index_to_char = build_vocab(train_text)
vocab_size = len(char_to_index)
context_length = 32

# Saving vocab to JSON for later use
vocab_path = os.path.join(parent_dir, "char_to_index.json")
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(char_to_index, f, ensure_ascii=False, indent=2)
print("Vocabulary saved to char_to_index.json")

dataset = CharDataset(train_text, char_to_index, context_length)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharacterTransformer(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# lets train the model now
model.train()
for epoch in range(3):
    total_loss = 0
    for x_batch, y_batch in data_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        
    print(f"Epoch {epoch+1}/3 - Avg Loss: {total_loss/len(data_loader):.4f}")

# Saving model state 
torch.save(model.state_dict(), "character_transformer.pt")
print("Model saved to character_transformer.pt")
