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

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    @classmethod
    def load_training_data(cls, train_dataset):
        """
        Normalizes and loads training data.
        """
        # Parse the conversation string into a list of dictionaries
        train_conversations = ast.literal_eval(train_dataset['conversations'])
        normalized_train_data = cls.normalize_conversations(train_conversations)
        return normalized_train_data

    @classmethod
    def load_test_data(cls, fname):
        """
        Loads and normalizes test data from the given file.
        """
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # remove the last character (newline)
                data.append(inp)
        # Normalize test data if needed
        return data

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
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            dummy_save = f.read()
        return MyModel()

    @staticmethod
    def normalize_value(text):
        """
        Normalizes a given text by removing unwanted characters and handling accents.
        """
        # Remove leading/trailing spaces
        text = text.strip()

        # Remove special characters and unwanted punctuation
        text = re.sub(r'[^A-Za-z0-9áéíóúàèìòùäëïöüâêîôûãõÇçÁÉÍÓÚ]+', ' ', text)

        # Normalize text: converting unicode characters (accents) to standard characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

        # Optionally, you can use a language detection library to handle language-specific normalizations
        try:
            lang = detect(text)
        except:
            lang = "unknown"

        return text, lang

    @staticmethod
    def normalize_conversations(conversation):
        """
        Normalize a list of conversation entries.
        """
        normalized = []
        for entry in conversation:
            text = entry["value"]
            normalized_text, lang = MyModel.normalize_value(text)
            normalized.append({
                "normalized": normalized_text,
                "language": lang
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

    # Load the dataset
    dataset = load_dataset("csv", data_files="src/output/mldd_dataset.csv")

    # Split the dataset into train and validation sets (90% train, 10% validation)
    train_dataset, dev_dataset = dataset["train"].train_test_split(test_size=0.1).values()

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = MyModel()
        print('Loading training data')
        train_data = model.load_training_data(train_dataset)
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = model.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
