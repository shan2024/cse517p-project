import os
import json
import re
from typing import List, Tuple, Dict, Union
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import importlib
from collections import Counter
import random

import sys
import os

import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from Transformer_Based.transformer_wrapper import TransformerModelWrapper
import torch
from helpers import load_test_input, write_pred
import time

from helpers import DatasetFileLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")


import random
import pandas as pd

def to_sample_and_expected_result(data, min_input_size = 3, max_input_size=300):
    """
    This function is used to build a test set in the example format so the model can be evaluated
    Splits the provided input into sample, y_true pairs.
    Each line is split at a random index between MIN_INPUT_SIZE and MAX_INPUT_SIZE
    y_true represents the next chracter after the split. The remainder of the string is discarded

    Output:

    x: list of training samples
    y_true: list of expected outputs
    """
    x = []
    y_true = []

    for i, line in enumerate(data):
        if line != "" and type(line) == str:
            
            words = line.split()

            splittable_words = []
            
            # We only split within words since we will not need to predict whitespace
            #Since we know we will get minimum one character, of the word and will not predict whitespace, we will only split words of at least 3 characters
            for i,word in enumerate(words):
                if len(word) >= 3:
                    splittable_words.append((word, i))

            if len(splittable_words) > 0:

                #Pick a random word to split
                split_word_index = random.randint(0, len(splittable_words)-1)
                
                #Get the index and the word to split
                to_split, og_index = splittable_words[split_word_index]

                split_index = random.randint(1, len(to_split)-1) #Get a spot to split. It cannot be first or last character
                
                #split the word
                splitted = to_split[0:split_index]

                # Record the expected y value
                y = to_split[split_index]

                # Split our words to only get ones that come before the split word
                res = words[0:og_index]

                #Add back in the word we split
                res.append(splitted)

                x.append(" ".join(res))
                y_true.append(y)
        

    return x, y_true

def write_test_data(test_data_dir, x_test, y_test):
    with open(f'{test_data_dir}/input.txt', 'w') as f:
        for item in x_test:
            f.write(f"{item}\n")

    with open(f'{test_data_dir}/answer.txt', 'w') as f:
        for item in y_test:
            f.write(f"{item}\n")

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_directory', help='Directory containing clean data', default='data/parsed_data')
    parser.add_argument('--test_data', help='path to write test data', default='test/')
    args = parser.parse_args()

    # Load combined data
    loader = DatasetFileLoader()
    loader.load(args.data_directory)
    loader.test_data.head()

    loader.test_data = loader.test_data.drop(index=0)

    test_data_by_language = {"english": ["test_nasa.csv", "test_trek.csv"], "spanish": ["test_culturax_spanish.csv"]}

    # Process test data for each language
    for language, csv_files in test_data_by_language.items():
        print(f"Processing {language} test data...")
        
        # Create language-specific directory
        language_dir = os.path.join(args.test_data, language)
        os.makedirs(language_dir, exist_ok=True)
        
        # Load and process each CSV file for the language
        for csv_file in csv_files:
            file_path = os.path.join(args.data_directory, csv_file)
            
            # Skip if file doesn't exist
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} does not exist. Skipping...")
                continue
                
            # Load the CSV file
            data_df = pd.read_csv(file_path)
            
            # Generate test samples
            x_test, y_test = to_sample_and_expected_result(data_df.iloc[:, 0])
            
            # Write the test data
            write_test_data(language_dir, x_test, y_test)
            print(f"  Wrote {len(x_test)} test samples for {csv_file}")
        
    # Also process the combined test data as before
    x_test, y_test = to_sample_and_expected_result(loader.test_data[0])
    write_test_data(args.test_data, x_test, y_test)
    print(f"Wrote {len(x_test)} test samples for combined data")



