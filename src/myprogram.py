#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datasets import load_dataset
import re
import unicodedata
from langdetect import detect
import ast
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import subprocess
import torch
import torch.nn as nn

# Ensure necessary NLTK data files are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class MyModel(nn.Module):
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    @classmethod
    def load_training_data(cls, train_dataset):
        """
        Normalizes and loads training data, splits each conversation into individual words.
        """
        try:
            train_conversations = train_dataset['conversations']
        except Exception as e:
            print(f"Error parsing conversations field: {e}")
            raise
        normalized_train_data = cls.normalize_conversations(train_conversations)
        return normalized_train_data

    @classmethod
    def load_dev_data(cls, dev_dataset):
        """
        Normalizes and loads dev data, splits each conversation into individual words.
        """
        try:
            dev_conversations = dev_dataset['conversations']
        except Exception as e:
            print(f"Error parsing conversations field: {e}")
            raise
        normalized_dev_data = cls.normalize_conversations(dev_conversations)
        return normalized_dev_data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        pass

    def run_pred(self, data):
        # your code here
        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            # this model just predicts a random character each time
            top_guesses = [random.choice(all_chars) for _ in range(3)]
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        """
        Save the model's state dict to the specified directory.
        """
        model_path = os.path.join(work_dir, 'model.pt')  # Save as a .pth file (PyTorch format)
        torch.save(self.state_dict(), model_path)  # Save only the model's state_dict
        print(f"Model saved to {model_path}")

    @classmethod
    def load(cls, work_dir):
        model_path = os.path.join(work_dir, 'model.pt')
        model = cls()  # Create a fresh instance
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()  # Set to evaluation mode if needed
        print(f"Model loaded from {model_path}")
        return model

    @staticmethod
    def normalize_value(text):
        """
        Normalizes a given text by removing unwanted characters, splitting into words,
        and preserving all Unicode characters (including non-Latin).
        """
        # Remove leading/trailing spaces
        text = text.strip()

        # Remove escaped newlines (\\n) completely
        text = text.replace('\\n', '')

        # Remove actual newlines (\n) completely
        text = text.replace('\n', '')

        # Use a regex pattern to keep only letters (including Unicode letters) and spaces
        text = re.sub(r'[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF\u4e00-\u9fff\uac00-\ud7af\s]+', '', text)

        # Tokenize the text into words using RegexpTokenizer (no need for punkt)
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(text)

        # Remove stopwords using NLTK's stopwords list TODO: more languages
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.lower() not in stop_words]

        return words

    @staticmethod
    def normalize_conversations(conversation):
        """
        Normalize a list of conversation entries. Each entry's value is tokenized into words.
        """
        normalized = []
        # Ensure that conversation is a list of dictionaries, if not, parse it
        if isinstance(conversation, str):
            try:
                conversation = ast.literal_eval(conversation)  # Parse the string into a list of dictionaries
            except ValueError as e:
                print("Error parsing conversation string:", e)
                return []

        for entry in conversation:
            if isinstance(entry, dict):  # If entry is a dictionary
                text = entry.get("value", "")  # Extract the 'value' field
            elif isinstance(entry, str):  # If entry is a string
                text = entry  # Use the string directly
            else:
                print("Error parsing conversation entry")
                return
            normalized_words = MyModel.normalize_value(text)
            normalized.append({
                "normalized": normalized_words  # List of words
            })
        return normalized


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    # Check if the mldd_dataset.csv file exists
    dataset_file = 'output/mldd_dataset.csv'
    if not os.path.isfile(dataset_file):
        print(f"{dataset_file} not found. Running the necessary script to generate it.")
        # Run the script from src/util/combine_dataset_files.py to combine the dataset splits
        script_path = 'src/util/combine_dataset_files.py'
        subprocess.run(['python', script_path], check=True)
    else:
        print(f"{dataset_file} found. Proceeding with loading the dataset.")
    
    # Load the dataset
    dataset = load_dataset("csv", data_files="output/mldd_dataset.csv")

    # Split the dataset into train and validation sets (90% train, 10% validation)
    train_dataset, dev_dataset = dataset["train"].train_test_split(test_size=0.1).values()


    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = MyModel()
        print('Normalizing training data...')
        normalized_train_data = MyModel.load_training_data(train_dataset)  # Normalize training data
        print('Training')
        model.run_train(normalized_train_data, args.work_dir)  # Train with normalized train data
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Normalizing test data...')
        normalized_dev_data = MyModel.load_dev_data(dev_dataset)  # Normalize dev data
        print('Making predictions')
        pred = model.run_pred(normalized_dev_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(normalized_dev_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
