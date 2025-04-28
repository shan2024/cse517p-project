#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm


class CustomDataset(Dataset):
    def __init__(self, raw_text_file, spm_model_path, seq_length=30, pad_token_id=0):
        self.seq_length = seq_length
        self.pad_token_id = pad_token_id  # Define a special token for padding

        self.spm = spm.SentencePieceProcessor()
        self.spm.load(spm_model_path)  # Load SentencePiece model

        with open(raw_text_file, "r", encoding="utf-8") as f:
            self.text_lines = [line.strip() for line in f if line.strip()]  # Filter out empty lines

    def __len__(self):
        return len(self.text_lines)

    def __getitem__(self, idx):
        # Get a line of text from the raw file
        text = self.text_lines[idx]

        # Tokenize using SentencePiece
        tokens = self.spm.encode(text, out_type=int)

        # Ensure the sequence is long enough for the input-output pair
        if len(tokens) < self.seq_length + 1:
            tokens = tokens + [self.pad_token_id] * (self.seq_length + 1 - len(tokens))

        # Create input-output pair
        input_tokens = torch.tensor(tokens[:self.seq_length], dtype=torch.long)  # Input sequence
        target_token = torch.tensor(tokens[self.seq_length], dtype=torch.long)  # Next token to predict

        return input_tokens, target_token


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        last_hidden_state = rnn_out[:, -1, :]  # Use the last hidden state
        output = self.fc(last_hidden_state)
        return output

    def train_model(self, dataset, batch_size, learning_rate, num_epochs=10, device='cpu'):
        # data should be a DataFrame with 'input' and 'target' columns
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.to(device)

        for epoch in range(num_epochs):
            for inputs, targets in data_loader:
                optimizer.zero_grad()
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
        return self

    def evaluate_model(self, data, device='cpu'):
        self.eval()
        self.to(device)
        with torch.no_grad():
            total_loss = 0
            for inputs, targets in data:
                outputs = self(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
        avg_loss = total_loss / len(data)
        print(f'Average Loss: {avg_loss}')
        return avg_loss

    def predict_model(self, data):
        self.eval()
        predictions = []
        with torch.no_grad():
            for inputs in data:
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted)
        return predictions
