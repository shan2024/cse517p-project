#!/usr/bin/env python
import os
import string
import random
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict, Counter


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self, max_n=5):
        self.max_n = max_n
        self.models = {}
        self.top_unigrams = []

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        with open("src/train.txt", "r", encoding="utf-8") as f:
            return f.read()

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        for n in range(1, self.max_n + 1):
            self.models[n] = defaultdict(Counter)
            for i in range(len(data) - n):
                context = data[i:i + n - 1] if n > 1 else ''
                next_char = data[i + n - 1]
                self.models[n][context][next_char] += 1

        # Compute top unigrams as fallback
        all_chars = Counter(data)
        self.top_unigrams = [c for c, _ in all_chars.most_common(10)]

    def run_pred(self, data):
        # your code here
        preds = []
        for context in data:
            preds.append(self.predict_next_chars(context))
        return preds

    def predict_next_chars(self, context, top_k=3):
        candidates = []
        seen = set()
        context = context or ''

        for n in range(self.max_n, 0, -1):
            ctx = context[-(n - 1):] if n > 1 else ''
            model = self.models.get(n, {})
            dist = model.get(ctx, {})
            sorted_chars = sorted(dist.items(), key=lambda x: x[1], reverse=True)
            for char, _ in sorted_chars:
                if char not in seen:
                    candidates.append(char)
                    seen.add(char)
                if len(candidates) >= top_k:
                    return ''.join(candidates[:top_k])

        # Fallback to top unigrams
        for char in self.top_unigrams:
            if char not in seen:
                candidates.append(char)
            if len(candidates) >= top_k:
                break

        return ''.join(candidates[:top_k])

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.pkl'), 'wb') as f:
            pickle.dump({
                "max_n": self.max_n,
                "models": self.models,
                "top_unigrams": self.top_unigrams
            }, f)

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.pkl'), 'rb') as f:
            obj = pickle.load(f)
        model = cls(max_n=obj['max_n'])
        model.models = obj['models']
        model.top_unigrams = obj['top_unigrams']
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
