import torch
import torch.nn.functional as F
import json
from character_transformer_model import CharacterTransformer
import os
import pandas as pd

# Load vocab which was saved before
vocab_path = os.path.join("src", "Transformer_Based", "char_to_index.json")
with open(vocab_path, "r", encoding="utf-8") as f:
    char_to_index = json.load(f)
index_to_char = {i: ch for ch, i in char_to_index.items()}
vocab_size = len(char_to_index)
context_length = 32

#load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharacterTransformer(vocab_size).to(device)
model.load_state_dict(torch.load("character_transformer.pt", map_location=device))
model.eval()

# check predictions
user_input = input("Enter a string: ").strip()
input_ids = [char_to_index.get(c, char_to_index[' ']) for c in user_input][-context_length:]
input_ids = [char_to_index[' ']] * (context_length - len(input_ids)) + input_ids
input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(input_tensor)[0]
    probs = F.softmax(logits, dim=0)
    top3 = torch.topk(probs, k=3)

print("\nTop-3 predicted next characters:")
for idx, prob in zip(top3.indices, top3.values):
    print(f"  '{index_to_char[idx.item()]}' with probability {prob.item():.4f}")
