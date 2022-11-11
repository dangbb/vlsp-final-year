from typing import List

import sumy
import sumy.nlp.tokenizers
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.utils import get_stop_words

from src.config.config import load_config_from_json
from src.loader.class_loader import Dataset
from src.model.model import Model

LANGUAGE = "vietnamese"


class TextrankRanker(Model):
    def __init__(self):
        super(TextrankRanker, self).__init__()
        self.LANGUAGE: str = "vietnamese"

        self.tokenizer: sumy.nlp.tokenizers.Tokenizer = Tokenizer(LANGUAGE)
        self.summarizer: sumy.summarizers.text_rank = TextRankSummarizer()
        self.summarizer.stop_words = get_stop_words(LANGUAGE)

    def training(self, dataset: Dataset) -> None:
        pass

    def __call__(self, sents: List[str]) -> List[float]:
        all_sents = sents
        parser: PlaintextParser = PlaintextParser.from_string('  .  '.join(all_sents), self.tokenizer)

        ratings = self.summarizer.rate_sentences(parser.document)

        assert len(ratings) == len(all_sents), \
            "N Sent in doc diff from N sent in cluster, {} != {}".format(
                len(ratings),
                len(all_sents),
            )

        return list(ratings.values())


if __name__ == "__main__":
    from src.loader.class_loader import load_cluster

    dataset = load_cluster(
        "/home/hvn/Documents/dskt/vlsp-final-year/dataset/vlsp_2022_abmusu_train_data_new.jsonl",
        1,
    )

    config = load_config_from_json()
    lxr = TextrankRanker()
    lxr.training(dataset)
    score = lxr(dataset.clusters[0].get_all_sents())

    print("Score 0: ", score)
