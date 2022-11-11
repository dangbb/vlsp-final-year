import logging
import math
from typing import List

import numpy as np

from src.config.config import ModelConfig, load_config_from_json
from src.loader.class_loader import Dataset, Cluster
from src.model.mmr_query import MMRSummarizerQuery
from src.model.ranker.lexrank_ranker import LexrankRanker
from src.model.ranker.textrank_ranker import TextrankRanker
from src.model.ranker.tf_idf_ranker import TFIDFRanker

configs = {
    "models": [
        {
            "name": "tfidf",
            "output": 1,
            "fields": [
                "tfidf_sentence_weight",
            ]
        },
        {
            "name": "textrank",
            "output": 1,
            "fields": [
                "textrank_weight"
            ]
        },
        {
            "name": "lexrank",
            "output": 1,
            "fields": [
                "lexrank_sentence_weight",
            ]
        }
    ]
}


class CombinationRanker:
    def __init__(self, config: ModelConfig):
        logging.warning("ComnbinationRanker-init: Start create a MMR Summarizer instance")
        self.config = config

        self.lexrank = LexrankRanker()
        self.textrank = TextrankRanker()
        self.tf_idf = TFIDFRanker()

        self.mmr = MMRSummarizerQuery(config)

        self.SENTENCES_COUNT = config.params

        logging.warning("ComnbinationRanker-init: Model created")

    def training(self, dataset: Dataset):
        self.lexrank.training(dataset)
        self.textrank.training(dataset)
        self.tf_idf.training(dataset)
        pass

    def get_score(
            self,
            sentences: List[str],
            debug: int = 0
    ):
        tfidf_score = self.tf_idf(sentences)
        textrank_score = self.textrank(sentences)
        lexrank_score = self.lexrank(sentences)

        tfidf_score = np.array(tfidf_score, dtype=float)
        textrank_score = np.array(textrank_score, dtype=float)
        lexrank_score = np.array(lexrank_score, dtype=float)

        tfidf_score = (tfidf_score - tfidf_score.min()) / (tfidf_score.max() - tfidf_score.min() + 0.001)
        textrank_score = (textrank_score - textrank_score.min()) / (textrank_score.max() - textrank_score.min() + 0.001)
        lexrank_score = (lexrank_score - lexrank_score.min()) / (lexrank_score.max() - lexrank_score.min() + 0.001)

        scores = {
            "tfidf": tfidf_score,
            "textrank": textrank_score,
            "lexrank": lexrank_score
        }

        final_scores = np.zeros((len(sentences)), dtype=float)

        if debug == 1:
            return scores

        for model in configs["models"]:
            for i in range(model["output"]):
                final_scores = final_scores + self.config.config[model["fields"][i]] * np.array(scores[model["name"]], dtype=float)

        return final_scores

    def predict(self, cluster: Cluster) -> (List[str], List[float]):
        final_sents = []

        sent_count = len(cluster.get_all_sents())
        for SENT_COUNT in self.SENTENCES_COUNT:
            if 0 <= SENT_COUNT < 1:
                sent_count = min(int(math.ceil(len(cluster.get_all_sents()) * SENT_COUNT)), sent_count)
            else:
                sent_count = min(int(SENT_COUNT), sent_count)

        for doc in cluster.documents:
            sents = doc.get_all_sents()

            scores = self.get_score(sents)

            if len(sents) >= sent_count:
                idx = np.argpartition(scores, -sent_count)[-sent_count:]
            else:
                idx = list(range(len(sents)))
            final_sents = final_sents + [sents[i] for i in idx]

        final_sents = [sent.replace("_", " ") for sent in final_sents]

        return self.mmr(final_sents, sent_count, cluster.get_all_anchor())


if __name__ == "__main__":
    from src.loader.class_loader import load_cluster

    CSOURCE = 'sent_splitted_token'

    dataset = load_cluster(
        "/home/dang/vlsp-final-year/dataset/vlsp_2022_abmusu_train_data_new.jsonl",
        1,
    )
    dataset.set_source(CSOURCE)

    config = load_config_from_json()
    lxr = CombinationRanker(config.models[15])
    lxr.training(dataset)
    sents, rates = lxr.predict(dataset.clusters[0])

    print("** Predicted sent: \n", sents)
    print("** Rating: ", len(rates))