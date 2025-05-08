import os
import torch
import json
import torch.nn.functional as F
from .character_transformer_model import CharacterTransformer

class TransformerModelWrapper:
    def __init__(self, vocab_file, model_file, device):
        """
        Load in trained model and vocab
        """
        #input size must be 32 characters. If it is more or less it is padded or truncated
        self.context_length = 32

        # Load vocab which was saved before
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.char_to_index = json.load(f)
        self.index_to_char = {i: ch for ch, i in self.char_to_index.items()}
        vocab_size = len(self.char_to_index)
        
        #load the model
        self.model = CharacterTransformer(vocab_size).to(device)
        self.model.load_state_dict(torch.load(model_file, map_location=device))
        self.model.eval()
        self.device = device

    def embed_string(self, input):
        input_ids = [self.char_to_index.get(c, self.char_to_index[' ']) for c in input][-self.context_length:]
        input_ids = [self.char_to_index[' ']] * (self.context_length - len(input_ids)) + input_ids
        return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)

    def predict_single(self, input: str):
        input_tensor = self.embed_string(input)

        with torch.no_grad():
            logits = self.model(input_tensor)[0]
            probs = F.softmax(logits, dim=0)
            top3 = torch.topk(probs, k=3)
            return ''.join(self.index_to_char[i.item()] for i in top3.indices)

    def embed_strings(self, input: list[str]):

        output = torch.zeros(len(input), self.context_length, dtype=torch.long).to(self.device)

        for i, line in enumerate(input):
            output[i] = self.embed_string(line)
        
        return output

    def predict(self, input: list[str]):
        input_tensor = self.embed_strings(input)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=0)
            top3 = torch.topk(probs, k=3)

            res = []
            for row in top3.indices:
                pred = ""
                for i in row:
                    pred+=self.index_to_char[i.item()]

                res.append(pred)


            return res
