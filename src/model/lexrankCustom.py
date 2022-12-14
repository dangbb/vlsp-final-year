import logging
import math
from typing import List

from lexrank import LexRank

import re

from src.config.config import ModelConfig, load_config_from_json
from src.loader.class_loader import Cluster, Dataset
from src.model.model import Model


class Lexrank(Model):
    def __init__(self, config: ModelConfig):
        super(Lexrank, self).__init__()
        self.config = config
        self.model = None

        self.SENTENCES_COUNT = config.params

    def training(self, dataset: Dataset) -> None:
        pass

    def predict(self, cluster: Cluster) -> (List[str], List[float]):
        all_sents = cluster.get_all_sents()

        self.model = LexRank(all_sents)

        sent_count = len(all_sents)
        for SENT_COUNT in self.SENTENCES_COUNT:
            if 0 <= SENT_COUNT < 1:
                sent_count = min(int(math.ceil(len(all_sents) * SENT_COUNT)), sent_count)
            else:
                sent_count = min(int(SENT_COUNT), sent_count)

        return self.model.get_summary(all_sents, sent_count), self.model.rank_sentences(all_sents, threshold=None, fast_power_method=False)


if __name__ == "__main__":
    from src.loader.class_loader import Cluster, load_cluster

    SOURCE = 'sent_splitted_token'

    dataset = load_cluster(
        "/home/dang/vlsp-final-year/dataset/vlsp_2022_abmusu_train_data_new.jsonl",
        1,
    )
    dataset.set_source(SOURCE)

    config = load_config_from_json()
    lxr = Lexrank(config.models[0])
    lxr.training(dataset)
    sents, rates = lxr.predict(dataset.clusters[0])

    print("** Predicted sent: \n", sents)
    print("** Rating: ", len(rates))
