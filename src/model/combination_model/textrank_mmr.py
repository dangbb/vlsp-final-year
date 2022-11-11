import sumy

from src.loader.class_loader import Dataset, Cluster
from src.model.mmr import MMRSummarizer

import math
from typing import List

import sumy.nlp.tokenizers
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.utils import get_stop_words

from src.config.config import ModelConfig, load_config_from_json, Config
from src.model.model import Model
from src.model.textrank import extracter

LANGUAGE = "vietnamese"
SENTENCES_COUNT = 2


class TextRank_MMR_Model(Model):
    def __init__(self, config: ModelConfig):
        super(TextRank_MMR_Model, self).__init__()
        self.config = config

        self.LANGUAGE: str = "vietnamese"
        self.COARSE_SENTENCES_COUNT = config.params
        self.SENTENCES_COUNT = config.net_params

        self.tokenizer: sumy.nlp.tokenizers.Tokenizer = Tokenizer(LANGUAGE)
        self.summarizer: sumy.summarizers.text_rank = TextRankSummarizer()
        self.summarizer.stop_words = get_stop_words(LANGUAGE)

        self.redundant_remover = MMRSummarizer(config)

    def training(self, dataset: Dataset) -> None:
        pass

    def predict(self, cluster: Cluster) -> (List[str], List[float]):
        all_sents = cluster.get_all_sents()
        parser: PlaintextParser = PlaintextParser.from_string('  .  '.join(all_sents), self.tokenizer)

        sent_count = len(parser.document.sentences)
        for SENT_COUNT in self.COARSE_SENTENCES_COUNT:
            if 0 <= SENT_COUNT < 1:
                sent_count = min(int(math.ceil(len(all_sents) * SENT_COUNT)), sent_count)
            else:
                sent_count = min(int(SENT_COUNT), sent_count)

        net_sent_count = len(parser.document.sentences)
        for SENT_COUNT in self.SENTENCES_COUNT:
            if 0 <= SENT_COUNT < 1:
                net_sent_count = min(int(math.ceil(len(all_sents) * SENT_COUNT)), net_sent_count)
            else:
                net_sent_count = min(int(SENT_COUNT), net_sent_count)

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

        coarse_sents, _ = extracter(all_sents, list(ratings.values()), sent_count), list(ratings.values())

        net_sents = self.redundant_remover(coarse_sents, net_sent_count)
        return net_sents


if __name__ == "__main__":
    from src.loader.class_loader import Cluster, load_cluster

    SOURCE = 'sent_splitted_token'

    dataset = load_cluster(
        "/home/hvn/Documents/dskt/vlsp-final-year/dataset/vlsp_2022_abmusu_train_data_new.jsonl",
        1,
    )
    dataset.set_source(SOURCE)

    config = load_config_from_json()
    textrank_mmr = TextRank_MMR_Model(config.models[7])
    sents, rates = textrank_mmr.predict(dataset.clusters[0])

    print("** Predicted sent: \n", sents)
    print("** Rating: ", len(rates))