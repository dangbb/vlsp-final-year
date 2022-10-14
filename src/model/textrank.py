from __future__ import absolute_import, annotations
from __future__ import division, print_function, unicode_literals

import math
from typing import List

import sumy.nlp.tokenizers
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.utils import get_stop_words

from src.config.config import Config
from src.model.model import Model

LANGUAGE = "vietnamese"
SENTENCES_COUNT = 2


class TextrankCustom(Model):
    def __init__(self, config: Config):
        super(TextrankCustom, self).__init__()
        self.config = config
        self.LANGUAGE: str = "vietnamese"

        self.SENTENCES_COUNT = config.model_config['params']

        self.tokenizer: sumy.nlp.tokenizers.Tokenizer = Tokenizer(LANGUAGE)
        self.summarizer: sumy.summarizers.text_rank = TextRankSummarizer()
        self.summarizer.stop_words = get_stop_words(LANGUAGE)

    def predict(self, cluster: Cluster) -> (List[str], List[float]):
        parser: PlaintextParser = PlaintextParser.from_string('  .  '.join(cluster.get_all_sents()), self.tokenizer)

        sent_count = len(parser.document.sentences)
        for SENT_COUNT in self.SENTENCES_COUNT:
            if 0 <= SENT_COUNT < 1:
                sent_count = min(int(math.ceil(len(parser.document.sentences) * SENT_COUNT)), sent_count)
            else:
                sent_count = min(int(SENT_COUNT), sent_count)

        if self.config.DEBUG:
            all_sents = cluster.get_all_sents()
            print(">> Print all short sent (len < 5)")
            for sent in all_sents:
                if len(sent) < 5:
                    print(sent)
            print(">> Print first difference")
            for i, sent in enumerate(all_sents):
                if sent != str(parser.document.sentences[0]):
                    print("Difference at pos: ", i, sent, parser.document.sentences[0])
                    break

        sentences, ratings = self.summarizer(parser.document, sent_count)

        assert len(parser.document.sentences) == len(cluster.get_all_sents()), \
            "N Sent in doc diff from N sent in cluster, {} != {}".format(
                len(parser.document.sentences),
                len(cluster.get_all_sents()),
            )
        assert len(parser.document.sentences) == len(ratings), \
            "N Sent in doc diff from rating length, {} != {}".format(
                len(parser.document.sentences),
                len(ratings)
            )

        sentences = [str(sent) for sent in sentences]
        return sentences, list(ratings.values())


if __name__ == '__main__':
    from src.loader.class_loader import Cluster, load_cluster

    SOURCE = 'sent_splitted_token'

    dataset = load_cluster(
        "/home/dang/vlsp-final-year/dataset/vlsp_2022_abmusu_train_data_new.jsonl"
    )
    dataset.set_source(SOURCE)

    textrank = TextrankCustom(8, .1)
    textrank.predict(dataset.clusters[0])