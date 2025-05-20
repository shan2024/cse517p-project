import torch
from torch.utils.data import DataLoader
import pandas as pd
import random
import json
import os
import numpy as np

from character_transformer_model import CharacterTransformer
from utils import build_vocab, CharDataset
from helpers import DatasetFileLoader
from torch.optim.lr_scheduler import LambdaLR

parent_dir = os.path.dirname(os.path.abspath("__file__"))
data_dir = os.path.join(parent_dir, "../../data")
test_dir = os.path.join(parent_dir, "../../test")

# Load training data
data_loader = DatasetFileLoader()
data_loader.load(f"{data_dir}/parsed_data", .05, .05, .05)

train_data = data_loader.train_data[0][1:]
dev_data = data_loader.dev_data[0][1:]

train_text = "\n".join(train_data)
dev_text = "\n".join(dev_data)

# Build vocabulary 
char_to_index, index_to_char = build_vocab(train_text)
vocab_size = len(char_to_index)
context_length = 32

# Saving vocab to JSON
vocab_path = os.path.join(parent_dir, "char_to_index.json")
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(char_to_index, f, ensure_ascii=False, indent=2)
print("Vocabulary saved to char_to_index.json")

# Prepare datasets
train_dataset = CharDataset(train_text, char_to_index, context_length)
dev_dataset = CharDataset(dev_text, char_to_index, context_length)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=10000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharacterTransformer(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

last_dev_loss = np.inf
dev_increased_epoch_count = 0

num_epochs = 5

# Define warm-up parameters
warmup_steps = 500

# AI generated code
# Lambda function for linear warm-up
def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0  # or add decay logic later

scheduler = LambdaLR(optimizer, lr_lambda)

# Training loop with dev loss evaluation
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Evaluate on dev set
    model.eval()
    total_dev_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in dev_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            total_dev_loss += loss.item()
            
    avg_dev_loss = total_dev_loss / len(dev_loader)

    if avg_dev_loss > last_dev_loss:
        dev_increased_epoch_count+=1
        print(f"Dev loss has increased: {dev_increased_epoch_count} so far")
    else:
        dev_increased_epoch_count=0

    last_dev_loss = avg_dev_loss
    
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Dev Loss: {avg_dev_loss:.4f}")

    # Save model after each epoch
    torch.save(model.state_dict(), f"work/character_transformer_epoch_{epoch+1}.pt")
    print(f"Model saved to character_transformer_{epoch+1}.pt")

    if dev_increased_epoch_count >= 3:
        print("Dev loss increased three times in a row. Exiting training loop")
        break

torch.save(model.state_dict(), f"character_transformer_epoch.pt")
print(f"Model saved to character_transformer.pt")