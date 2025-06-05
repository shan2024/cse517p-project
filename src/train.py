#!/usr/bin/env python
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from Transformer_Based.transformer_wrapper import TransformerModelWrapper
import torch
import random
import time

if __name__ == '__main__':
    
    random.seed(42)
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--data_dir', help='where to load train data', default='work')
    parser.add_argument('--data_fraction', help='Fraction of the training data to train on', default=1)
    parser.add_argument('--time', help='Measure training time', action='store_true')
    parser.add_argument('--continue_training', help='Continue to train an exisiting model', action='store_true')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerModelWrapper(device, args.work_dir)

    if args.time:
        start_time = time.time()

    model.train(args.data_dir, continue_training=args.continue_training, dataset_fraction= float(args.data_fraction))

    if args.time:
        elapsed_time = time.time() - start_time
        print(f"Total training time: {elapsed_time:.2f} seconds")