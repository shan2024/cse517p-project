#!/usr/bin/env python
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from Transformer_Based.transformer_wrapper import TransformerModelWrapper
import torch
import random


if __name__ == '__main__':
    
    random.seed(42)
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--data_dir', help='where to load train data', default='work')
    parser.add_argument('--data_fraction', help='Fraction the training data to train on', default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerModelWrapper(device, args.work_dir)

    model.train(args.data_dir, float(args.data_fraction))