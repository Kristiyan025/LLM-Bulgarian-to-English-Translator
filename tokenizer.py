import os
from collections import defaultdict
import re

from parameters import vocab_size
from parameters import ind2tokenFileName, token2indFileName, mergesFileName
from utils import startToken, endToken, unkToken, padToken, transToken, startTokenIdx, transTokenIdx, endTokenIdx
from utils import load_data, read_corpus


class BPETokenizer:
    SPACE_SYMBOLS = ['\t', ' ']

    NUMERIC_SYMBOLS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    PUNCTUATION_SYMBOLS = [
        '!', '"', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
        ':', ';', '<', '=', '?', '@', '[', ']', '_', '”', '•', '€', '№', '≤'
    ]

    NON_COMBINABLE_SYMBOLS = NUMERIC_SYMBOLS + PUNCTUATION_SYMBOLS

    def __init__(self):
        self.ind2token = []
        self.token2ind = {}
        self.merges = {}

    @staticmethod
    def from_files():
        if not (os.path.exists(ind2tokenFileName)
                and os.path.exists(token2indFileName)
                and os.path.exists(mergesFileName)):
            raise RuntimeError("Tokenizer files missing.")

        tokenizer = BPETokenizer()
        tokenizer.ind2token = load_data(ind2tokenFileName)
        tokenizer.token2ind = load_data(token2indFileName)
        tokenizer.merges = load_data(mergesFileName)
        return tokenizer

    def pretokenize(self, corpus):
        symbols = list(sorted(set([symbol for line in corpus for symbol in line])))
        # print(symbols)

        self.ind2token = [startToken, endToken, unkToken, padToken, transToken] + symbols
        self.token2ind = {token: ind for ind, token in enumerate(self.ind2token)}

    def compress_corpus(self, corpus):
        pattern = f"[{re.escape(''.join(self.SPACE_SYMBOLS))}]"

        corpus_word_split = [re.split(pattern, "".join(line)) for line in corpus]
        words = [word for line in corpus_word_split for word in line]

        word_freqs = defaultdict(int)
        for word in words:
            word_freqs[word] += 1

        return word_freqs

    def calculate_initial_word_splits(self, word_freqs):
        return {word: [c for c in word] for word in word_freqs.keys()}

    def compute_pair_freqs(self, splits, word_freqs):
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for pair in zip(split[:-1], split[1:]):
                if not (pair[0] in self.NON_COMBINABLE_SYMBOLS or
                        pair[1] in self.NON_COMBINABLE_SYMBOLS):
                    pair_freqs[pair] += freq
        return pair_freqs

    def get_best_pair(self, pair_freqs):
        return max(pair_freqs.items(), key=lambda item: item[1])

    def merge_pair(self, pair, splits):
        new_splits = {}
        pair_str = ''.join(pair)
        for word in splits.keys():
            split = splits[word]
            if len(split) == 1:
                new_splits[word] = split
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [pair_str] + split[i + 2:]
                i += 1

            new_splits[word] = split

        return new_splits

    def train(self, corpus, log: bool = False):
        self.pretokenize(corpus)
        word_freqs = self.compress_corpus(corpus)
        splits = self.calculate_initial_word_splits(word_freqs)

        while len(self.ind2token) < vocab_size:
            pair_freqs = self.compute_pair_freqs(splits, word_freqs)
            best_pair, best_freq = self.get_best_pair(pair_freqs)
            best_pair_str = ''.join(best_pair)

            splits = self.merge_pair(best_pair, splits)
            self.merges[best_pair] = best_pair_str

            self.token2ind[best_pair_str] = len(self.ind2token)
            self.ind2token += [best_pair_str]

            if log and len(self.ind2token) % 100 == 0:
                print("Current Vocab Size:", len(self.ind2token))

    def tokenize(self, text):
        i = 0
        while i < len(text) - 1:
            if (text[i], text[i + 1]) in self.merges:
                text = text[:i] + [self.merges[(text[i], text[i + 1])]] + text[i + 2:]
                if i > 0:
                    i -= 1
            else:
                i += 1
        result = [self.token2ind.get(token, unkToken) for token in text]
        return result

    def guarded_tokenize(self, text):
        for token in (startToken, transToken):
            i = 0
            while i < len(text) - len(token) + 1:
                if token == "".join(text[i : i + len(token)]):
                    text = text[:i] + [token] + text[i + len(token):]
                i += 1
        tokenized = self.tokenize(text)

        for tokenIdx in (startTokenIdx, transTokenIdx):
            if tokenized.count(tokenIdx) > 1:
                raise ValueError("Multiple start/translations tokens found in the same sentence.")
        return tokenized

    def tokenize_corpus(self, source_filename, target_filename):
        source = read_corpus(source_filename)
        target = read_corpus(target_filename)

        tokenized_source = [self.tokenize(sentence) for sentence in source]
        tokenized_target = [self.tokenize(sentence) for sentence in target]

        tokenized_corpus = [[startTokenIdx] + s + [transTokenIdx] + t + [endTokenIdx]
                            for (s,t) in zip(tokenized_source, tokenized_target)]
        return tokenized_corpus

    def untokenize(self, tokens):
        return [self.ind2token[token] for token in tokens]


if __name__ == "__main__":
    tokenizer = BPETokenizer.from_files()

    print(tokenizer.tokenize(list("Ето защо приветствам всяко действие, което допринася за насърчаване на мира, стабилността и принципите на правовата държава в страни и региони, намиращи се в криза.")))
