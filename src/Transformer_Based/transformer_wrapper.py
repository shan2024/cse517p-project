import os
import torch
import json
import numpy as np
import torch.nn.functional as F
from .character_transformer_model import CharacterTransformer
from .vocab import init_vocab
from .character_dataset import CharDatasetWrapper
from torch.utils.data import DataLoader
import multiprocessing
from torch.amp import autocast, GradScaler 
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from typing import Final
import time
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR


class TransformerModelWrapper:
    def __init__(self, device, work_directory, use_existing_vocab=True):
        """
        Load in trained model and vocab
        """
        # The model's files will be saved and loaded from this directory
        self.work_directory = work_directory

        #input size must be 32 characters. If it is more or less it is padded or truncated
        self.context_length = 32
        self.device = device
        self.vocab_file_name = "char_to_index.json"
        self.model_file_name = "character_transformer.pt"
        
        self.model_file_path = os.path.join(work_directory, self.model_file_name)
        
        # Set up vocab
        vocab_file_path = os.path.join(work_directory, self.vocab_file_name)

        # If the vocab file already exists then we should load it in
        if os.path.exists(vocab_file_path) and use_existing_vocab:
            with open(vocab_file_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            self.vocab = init_vocab(vocab_file_path)

        self.PAD_TOKEN: Final = self.vocab[' ']

        self.index_to_char = np.array(list(self.vocab.keys()))

        # Set up the model
        # We need to hold onto both the compiled and non compiled models.
        # We will use the compiled model except when saving and loading
        self.model = CharacterTransformer(len(self.vocab)).to(self.device)

        if hasattr(torch, 'compile'):
            self.compiled_model = torch.compile(self.model)
        
    def get_vocab_size(self):
        return len(self.vocab)
    
    def load(self):
        
        self.model.load_state_dict(torch.load(self.model_file_path, map_location=self.device))

        if hasattr(torch, 'compile'):
            self.compiled_model = torch.compile(self.model)
       
    def embed_strings(self, inputs: list[str]):

        encoded = np.full((len(inputs), self.context_length), 0, dtype=np.int32)

        for i, s in enumerate(inputs):
            indices = [self.vocab.get(c, self.PAD_TOKEN) for c in s[-self.context_length:]]

            if len(indices) < self.context_length:
                indices = [self.PAD_TOKEN] * (self.context_length - len(indices)) + indices

            encoded[i] = indices

        # Convert once to torch tensor and move to GPU
        return torch.from_numpy(encoded).to(self.device)


    def predict(self, input: list[str]):
        self.model.eval().half()
        input_tensor = self.embed_strings(input)

        with torch.no_grad():
            #TODO: Compiled model is slower. Determine if this is still the case with larger models
            logits = self.model(input_tensor)

            logits[:, self.PAD_TOKEN] = float('-inf')
            top3 = torch.topk(logits, k=3, dim=1).indices.cpu().tolist()

            res = ["".join(self.index_to_char[j] for j in row) for row in top3]
            
        return res
    
    def train(self, dataset: CharDatasetWrapper, num_epochs: int = 3, lr: float = 1e-4, batch_size=1048, verbose=True, save_checkpoints=False):

        #Create the checkpoints folder if it doesn't exist
        if save_checkpoints:
            os.makedirs(os.path.join(self.work_directory, "checkpoints"), exist_ok=True)
            self.model_checkpoint_path = f"{self.work_directory}/checkpoints/{self.model_file_name}"

        # Prepare datasets
        num_workers = min(20, multiprocessing.cpu_count() - 2)
        train_loader = DataLoader(dataset.train_dataset(), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers,  prefetch_factor=2, persistent_workers=True)
        dev_loader = DataLoader(dataset.dev_dataset(), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers,  prefetch_factor=2, persistent_workers=True)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        scaler = GradScaler()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=0.1,
            anneal_strategy="cos",
            final_div_factor=1e4,
        )

        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0

            # Add progress bar
            for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()

                # AMP forward
                with autocast(device_type=self.device.type):
                    logits = self.compiled_model(x_batch)
                    loss = self.loss_fn(logits, y_batch)

                # AMP backward
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.compiled_model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
                scheduler.step()

                scaler.unscale_(self.optimizer)

                if verbose:
                    total_train_loss += loss.item()

            if verbose:
                avg_train_loss = total_train_loss / len(train_loader)
                dev_loss = self.eval_loss(dev_loader)
                print(f"[train] Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}. Dev Loss: {dev_loss:.4f}")

            if save_checkpoints:
                torch.save(self.model.state_dict(), f"{self.model_checkpoint_path}.{epoch}")
        
        torch.save(self.model.state_dict(), self.model_file_path)
        print(f"[train] Model saved to {self.model_file_path}")
    
    def eval_loss(self, dataloader: DataLoader):
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