from typing import Union, Tuple, List, Iterable, Dict
import collections
import string
import os
import json
from .WordTokenizer import WordTokenizer, ENGLISH_STOP_WORDS
from vncorenlp import VnCoreNLP
import operator
from functools import reduce
class VietnameseTokenizer(WordTokenizer):
    """
    Simple and fast white-space tokenizer. Splits sentence based on white spaces.
    Punctuation are stripped from tokens.
    """
    def __init__(self, vocab: Iterable[str] = [], stop_words: Iterable[str] = ENGLISH_STOP_WORDS, do_lower_case: bool = False, vncorenlp_path = None):
        self.stop_words = set(stop_words)
        self.do_lower_case = do_lower_case
        self.set_vocab(vocab)
        self.vncorenlp_path = vncorenlp_path
        self.rdrsegmenter = VnCoreNLP(vncorenlp_path, annotators="wseg", max_heap_size='-Xmx1g')

    def get_vocab(self):
        return self.vocab

    def set_vocab(self, vocab: Iterable[str]):
        self.vocab = vocab
        self.word2idx = collections.OrderedDict([(word, idx) for idx, word in enumerate(vocab)])

    def segment(self, text: str) -> str:
        ''' Segment words in text and then flat the list '''
        segmented_word = self.rdrsegmenter.tokenize(text)
        return ' '.join(reduce(operator.concat, segmented_word))

    def tokenize(self, text: str) -> List[int]:
        #segment words in text
        text = self.segment(text)
        
        if self.do_lower_case:
            text = text.lower()

        tokens = text.split()

        tokens_filtered = []
        for token in tokens:
            if token in self.stop_words:
                continue
            elif token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

            token = token.strip(string.punctuation)
            if token in self.stop_words:
                continue
            elif len(token) > 0 and token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

            token = token.lower()
            if token in self.stop_words:
                continue
            elif token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue
            tokens_filtered.append(0)
        return tokens_filtered

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'VietnameseTokenizer_config.json'), 'w') as fOut:
            json.dump({'vocab': list(self.word2idx.keys()), 'stop_words': list(self.stop_words), 'do_lower_case': self.do_lower_case, 'vncorenlp_path': self.vncorenlp_path}, fOut)

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'VietnameseTokenizer_config.json'), 'r') as fIn:
            config = json.load(fIn)

        return VietnameseTokenizer(**config)
        