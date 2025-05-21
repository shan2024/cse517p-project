import os
import torch
import json
import numpy as np
import torch.nn.functional as F
from .character_transformer_model import CharacterTransformer
from .character_dataset import CharDatasetWrapper, build_vocab, CharDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

class TransformerModelWrapper:
    def __init__(self, device, work_directory):
        """
        Load in trained model and vocab
        """

        print(device)
        # The model's files will be saved and loaded from this directory
        self.work_directory = work_directory

        #input size must be 32 characters. If it is more or less it is padded or truncated
        self.context_length = 32
        self.device = device
        self.vocab_file_name = "char_to_index.json"
        self.model_file_name = "character_transformer.pt"

        self.vocab_file_path = os.path.join(work_directory, self.vocab_file_name)
        self.model_file_path = os.path.join(work_directory, self.model_file_name)
        self.model_checkpoint_path = f"{work_directory}/checkpoints/{self.model_file_name}"

        self.index_to_char = None
        self.char_to_index = None

    def load(self):
        # Load vocab which was saved before
        with open(self.vocab_file_path, "r", encoding="utf-8") as f:
            self.char_to_index = json.load(f)
        self.index_to_char = {i: ch for ch, i in self.char_to_index.items()}
        vocab_size = len(self.char_to_index)
        
        #load the model
        self.model = CharacterTransformer(vocab_size).to(self.device)
        self.model.load_state_dict(torch.load(self.model_file_path, map_location=self.device))
        self.model.eval()

    def embed_string(self, input):
        input_ids = [self.char_to_index.get(c, self.char_to_index[' ']) for c in input][-self.context_length:]
        input_ids = [self.char_to_index[' ']] * (self.context_length - len(input_ids)) + input_ids
        return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def embed_strings(self, input: list[str]):
        output = torch.zeros(len(input), self.context_length, dtype=torch.long).to(self.device)
        for i, line in enumerate(input):
            output[i] = self.embed_string(line)
        
        return output

    def predict_single(self, input: str):
        input_tensor = self.embed_string(input)

        with torch.no_grad():
            logits = self.model(input_tensor)[0]
            probs = F.softmax(logits, dim=0)
            top3 = torch.topk(probs, k=3)
            return ''.join(self.index_to_char[i.item()] for i in top3.indices)

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
    
    def train(self, data_directory, dataset_fraction: float=1.0):

        # TODO: Pass in the hyperparameters
        num_epochs = 5

        dataset = CharDatasetWrapper(self.device, data_directory, self.context_length, dataset_fraction)
        
        # Write the vocab to a file
        with open(self.vocab_file_path, "w", encoding="utf-8") as f:
            json.dump(dataset.vocab(), f, ensure_ascii=False, indent=2)

        # Prepare datasets
        train_loader = DataLoader(dataset.train_dataset(), batch_size=64, shuffle=True, pin_memory=True, num_workers=22)
        #dev_loader = DataLoader(dataset.dev_dataset(), batch_size=10000)

        self.model = CharacterTransformer(dataset.vocab_size()).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Training loop with dev loss evaluation
        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = self.loss_fn(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")

            # Save model after each epoch
            torch.save(self.model.state_dict(), f"{self.model_checkpoint_path}.{epoch}")

        torch.save(self.model.state_dict(), self.model_file_path)
        print(f"Model saved to character_transformer.pt")

    def eval_perplexity(self, dataloader: DataLoader):
        self.model.eval()
        total_dev_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(x_batch)
                loss = self.loss_fn(logits, y_batch)
                total_dev_loss += loss.item()
                
        avg_dev_loss = total_dev_loss / len(dataloader)
        return avg_dev_loss