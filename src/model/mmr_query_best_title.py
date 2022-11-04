import logging
import math
from typing import List

import numpy as np

from src.config.config import ModelConfig, load_config_from_json
from src.evaluate.rouge_evaluator import PipRouge
from src.loader.class_loader import Cluster
from src.model.mmr_query import MMRSummarizerQuery
from src.model.model import Model
from src.utils.embedding import Embedding, get_embedding
from src.utils.similarity import get_similarity


class MMRQueryBestTitle(Model):
    def __init__(self, config: ModelConfig):
        super(MMRQueryBestTitle, self).__init__()
        self.embedding: Embedding = get_embedding(config.embedding)
        self.similarity = get_similarity(config.embedding)
        self.summarizer = MMRSummarizerQuery(config)

        self.SENTENCES_COUNT = config.params

    def predict(self, cluster: Cluster) -> (List[str], List[float]):
        all_sents = cluster.get_all_sents()

        sent_count = len(all_sents)
        for SENT_COUNT in self.SENTENCES_COUNT:
            if 0 <= SENT_COUNT < 1:
                sent_count = min(int(math.ceil(len(all_sents) * SENT_COUNT)), sent_count)
            else:
                sent_count = min(int(SENT_COUNT), sent_count)

        rouge = PipRouge()
        best_score = -1
        best_title = ""
        for title in cluster.get_all_title():
            try:
                rouge_score = rouge(title, '.'.join(cluster.get_all_sents()))
                if rouge_score.rouge2.p > best_score:
                    best_score = rouge_score.rouge2.p
                    best_title = title
            except Exception as e:
                print("Failed title: ", title)

        sentences, scores = self.summarizer(all_sents, sent_count, [best_title])

        return sentences, scores


if __name__ == '__main__':
    from src.loader.class_loader import Cluster, load_cluster

    SOURCE = 'sent_splitted_token'

    dataset = load_cluster(
        "/home/dang/vlsp-final-year/dataset/vlsp_2022_abmusu_train_data_new.jsonl",
        1,
    )
    dataset.set_source(SOURCE)

    config = load_config_from_json()
    mmr = MMRQueryBestTitle(config.models[0])
    sents, rates = mmr.predict(dataset.clusters[0])

    print("** Predicted sent: \n", sents)
    print("** Rating: ", len(rates))
