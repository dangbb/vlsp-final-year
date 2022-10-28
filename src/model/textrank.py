from __future__ import absolute_import, annotations
from __future__ import division, print_function, unicode_literals

import math
from typing import List

import sumy.nlp.tokenizers
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.utils import get_stop_words

from src.config.config import Config, ModelConfig, load_config_from_json
from src.model.model import Model

LANGUAGE = "vietnamese"
SENTENCES_COUNT = 2


def extracter(sents: List[str], scores: List[float], n_sent: int) -> List[str]:
    assert len(sents) == len(scores), "Sent and score len are difference, {} != {}".format(
        len(sents),
        len(scores)
    )

    if n_sent > len(sents):
        n_sent = len(sents)

    group = [(sents[i], scores[i]) for i in range(len(sents))]

    group = sorted(group, key=lambda x: x[1], reverse=True)

    return [group[i][0] for i in range(n_sent)]


class TextrankCustom(Model):
    def __init__(self, config: ModelConfig):
        super(TextrankCustom, self).__init__()
        self.config = config
        self.LANGUAGE: str = "vietnamese"

        self.SENTENCES_COUNT = config.params

        self.tokenizer: sumy.nlp.tokenizers.Tokenizer = Tokenizer(LANGUAGE)
        self.summarizer: sumy.summarizers.text_rank = TextRankSummarizer()
        self.summarizer.stop_words = get_stop_words(LANGUAGE)

    def predict(self, cluster: Cluster) -> (List[str], List[float]):
        all_sents = cluster.get_all_sents()
        parser: PlaintextParser = PlaintextParser.from_string('  .  '.join(all_sents), self.tokenizer)

        sent_count = len(parser.document.sentences)
        for SENT_COUNT in self.SENTENCES_COUNT:
            if 0 <= SENT_COUNT < 1:
                sent_count = min(int(math.ceil(len(parser.document.sentences) * SENT_COUNT)), sent_count)
            else:
                sent_count = min(int(SENT_COUNT), sent_count)

        _, ratings = self.summarizer(parser.document, sent_count)

        assert len(parser.document.sentences) == len(all_sents), \
            "N Sent in doc diff from N sent in cluster, {} != {}".format(
                len(parser.document.sentences),
                len(all_sents),
            )
        assert len(parser.document.sentences) == len(ratings), \
            "N Sent in doc diff from rating length, {} != {}".format(
                len(parser.document.sentences),
                len(ratings)
            )

        return extracter(all_sents, list(ratings.values()), sent_count), list(ratings.values())


if __name__ == '__main__':
    from src.loader.class_loader import Cluster, load_cluster

    SOURCE = 'sent_splitted_token'

    dataset = load_cluster(
        "/home/Documents/dskt/hvn/vlsp-final-year/dataset/vlsp_2022_abmusu_train_data_new.jsonl",
        1,
    )
    dataset.set_source(SOURCE)

    config = load_config_from_json()
    textrank = TextrankCustom(config.models[0])
    sents, rates = textrank.predict(dataset.clusters[0])

    print("** Predicted sent: \n", sents)
    print("** Rating: ", len(rates))
