import torch
import torch.nn.functional as F
import pandas as pd
from character_transformer_model import CharacterTransformer
from utils import build_vocab

# Load training data from  CSVs
df1 = pd.read_csv("../../data/parsed_data/train_nasa.csv")
df2 = pd.read_csv("../../data/parsed_data/train_trek.csv")
combined = pd.concat([df1["dialogue"], df2["dialogue"]], ignore_index=True)
text = "\n".join(combined.dropna().astype(str).tolist())

# Build vocabulary
stoi, itos = build_vocab(text)
vocab_size = len(stoi)
context_len = 32

# Load model
model = CharacterTransformer(vocab_size)
model.load_state_dict(torch.load("character_transformer.pt", map_location="cpu"))
model.eval()
model.to("cpu")

# Get user input
input_text = input("Enter a string: ").strip()
encoded = [stoi.get(c, stoi[' ']) for c in input_text]

# Pad/truncate input
if len(encoded) < context_len:
    encoded = [stoi[' ']] * (context_len - len(encoded)) + encoded
else:
    encoded = encoded[-context_len:]

input_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

# Predict next character
with torch.no_grad():
    logits = model(input_tensor)[0]
    probs = F.softmax(logits, dim=0)
    top3 = torch.topk(probs, k=3)

# Output results
print("\nTop-3 predicted next characters:")
for i, p in zip(top3.indices, top3.values):
    print(f"  '{itos[i.item()]}' with probability {p.item():.4f}")
