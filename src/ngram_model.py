import os

from utils.normalize import normalize
from utils.constants import MAX_NGRAM_SIZE, MAX_UNIGRAM_FALLBACK_SIZE, MAX_TOP_K
from collections import defaultdict, Counter
import pickle

# NOTE: this is just a raw n-gram model with frequencies.
# It does not have smoothening, probabilities, weights (for each level) etc. It is a very basic Ngram model
class NGramModel:

    def __init__(self, max_grams = MAX_NGRAM_SIZE):
        self.max_grams = MAX_NGRAM_SIZE

        # Captures all the models (Fallbacks, n-1 ... 1)
        self.models = {}

        # Top unigrams will help in identifying the top if we don't find next available characters.
        self.top_unigrams = []

    @classmethod
    def load_training_data(cls):
        with open("data/en.txt", "r", encoding="utf-8") as f:
            return f.read()

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    '''
    Trains character level n-gram model from raw text.
    Builds models for all n from 1 to max_grams. Example 1-gram, 2-gram ... nGram.
    
    Example if max_grams = 3 for context "hello ":
    1 gram: context '', next_char = h
    2 gram: context 'h', next_char = e
    3 gram: context 'he', next_char = l
    
    Optimizations that can be done:
    1. We can do a word level n-gram and do a mix and match whenever we encounter a blank space (akashkg@)
    
    TODO: Try moving this to its own python script for better management.
    '''
    def run_train(self, raw_data, work_dir):

        # cleaning the raw data by normalizing it.
        # Basic normalization only
        # TODO: Do de-tokenization of the data - the data set always show ' ' after ' character.
        data =  normalize(raw_data)

        # Build the n-gram model upto max_grams - n
        # n = 1, 2... max_grams
        for n in range(1, self.max_grams + 1):

            # create the NGramTable
            self.models[n] = defaultdict(Counter)
            for i in range(len(data) - n):

                # Extract the context and the next character
                '''
                For internal & future reference:
                
                    n = 3, data = "hello "
                    at i - 0, context = he, next char = l
                    at i = 1, context = el, next char = l
                    at i = 2, context = ll, next char = 0
                    
                Also, if n is 0 (unigram), we default context to '', we look at the probability of each character.
                '''
                context = data[i:i + n - 1] if n > 1 else ''
                next_char = data[i + n - 1]

                # Updating the count of next char following this context
                self.models[n][context][next_char] += 1

        # Identify top unigrams - ignores frequency
        all_chars = Counter(data)
        self.top_unigrams = [char for char, _ in all_chars.most_common(MAX_UNIGRAM_FALLBACK_SIZE)]

    def run_pred(self, data):
        # your code here
        preds = []
        for context in data:
            preds.append(self.predict_next_chars(context))
        return preds

    def predict_next_chars(self, context, top_k=MAX_TOP_K):
        candidates = []
        seen = set()
        context = normalize(context)

        # Iterate from the max_grams to lower ngrams if context not found n ... 3, 2, 1
        for n in range(self.max_grams, 0, -1):
            # Returns the last (n-1) characters of the context; '' for unigram.
            ctx = context[-(n - 1):] if n > 1 else ''
            model = self.models.get(n, {})
            dist = model.get(ctx, {})
            sorted_chars = sorted(dist.items(), key=lambda x: x[1], reverse=True)
            for char, _ in sorted_chars:
                if char not in seen:
                    candidates.append(char)
                    seen.add(char)
                # if the condition is met.
                if len(candidates) >= top_k:
                    return ''.join(candidates[:top_k])

        # Fallback to top unigrams
        # In this case we would have had lesser characters than K
        for char in self.top_unigrams:
            if char not in seen:
                candidates.append(char)
            if len(candidates) >= top_k:
                break

        return ''.join(candidates[:top_k])

    def save(self, work_dir):

        with open(os.path.join(work_dir, 'model.sda'), 'wb') as f:
            pickle.dump({
                "max_n": self.max_grams,
                "models": self.models,
                "top_unigrams": self.top_unigrams
            }, f)

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.sda'), 'rb') as f:
            obj = pickle.load(f)
        model = cls(max_grams=obj['max_n'])
        model.models = obj['models']
        model.top_unigrams = obj['top_unigrams']
        return model