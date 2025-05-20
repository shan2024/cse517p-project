#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from Transformer_Based.transformer_wrapper import TransformerModelWrapper
from Transformer_Based.character_transformer_model import CharacterTransformer
import torch
from helpers import load_test_input, write_pred

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='test/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parent_dir = os.path.dirname(os.path.abspath("__file__"))

    random.seed(0)

    vocab_file = os.path.join(parent_dir, "src/Transformer_Based/char_to_index.json")
    model_file = os.path.join(parent_dir, "src/Transformer_Based/character_transformer.pt")

    model = TransformerModelWrapper(vocab_file, model_file, device)

    if args.mode == "train":
        model.train(.01)

    if args.mode == "test":

        test_data_file = args.test_data
        output_file = args.test_output

        test_input = load_test_input(test_data_file)

        preds = model.predict(test_input)

        write_pred(preds, output_file)
    

    