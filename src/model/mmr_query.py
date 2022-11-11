import logging
import math
from typing import List

import numpy as np

from src.config.config import ModelConfig, load_config_from_json
from src.loader.class_loader import Cluster, SOURCE
from src.model.model import Model
from src.utils.embedding import Embedding, get_embedding
from src.utils.similarity import get_similarity


class MMRSummarizerQuery:
    def __init__(self, config: ModelConfig):
        logging.warning("MMR-init: Start create a MMR Summarizer instance")
        self.config = config
        self.similarity = get_similarity(config.embedding)
        self.embedding = get_embedding(config.embedding)
        logging.warning("MMR-init: Model created")

    def __call__(
            self,
            sentences: List[str],
            n_sent: int,
            title: List[str] = None,
    ):
        self.embedding.fit(sentences)

        chosen_sentences = []
        unchosen_sentences = [i for i in range(len(sentences))]
        embedding_vector = self.embedding.transform(sentences)

        embedding_document = self.embedding.transform(['.'.join(title)])[0]

        sigma = self.config.sigma
        scores = np.zeros(len(sentences))

        while len(chosen_sentences) < n_sent:
            best_score = -1.0
            best_sent_idx = -1
            for i in unchosen_sentences:
                salient = self.similarity(
                    embedding_vector[i],
                    embedding_document,
                )
                if len(chosen_sentences) == 0:
                    redundancy = 0
                else:
                    redundancy = max([
                        self.similarity(
                            embedding_vector[i],
                            embedding_vector[j]
                        ) for j in chosen_sentences
                    ])
                score = salient * sigma - redundancy * (1 - sigma)
                if score > best_score:
                    best_score = score
                    best_sent_idx = i

            if best_sent_idx == -1:
                break
            chosen_sentences.append(best_sent_idx)
            unchosen_sentences.remove(best_sent_idx)
            scores[best_sent_idx] = 1.0

        return [sentences[i] for i in chosen_sentences], scores.tolist()


class MMRQuery(Model):
    def __init__(self, config: ModelConfig):
        super(MMRQuery, self).__init__()
        self.embedding: Embedding = get_embedding(config.embedding)
        self.similarity = get_similarity(config.embedding)
        self.summarizer = MMRSummarizerQuery(config)

        self.SENTENCES_COUNT = config.params

    def predict(self, cluster: Cluster) -> (List[str], List[float]):
        cluster.set_source(SOURCE.SENT_SPLITTED_TEXT.value)
        all_sents = cluster.get_all_sents()

        sent_count = len(all_sents)
        for SENT_COUNT in self.SENTENCES_COUNT:
            if 0 <= SENT_COUNT < 1:
                sent_count = min(int(math.ceil(len(all_sents) * SENT_COUNT)), sent_count)
            else:
                sent_count = min(int(SENT_COUNT), sent_count)

        sentences, scores = self.summarizer(all_sents, sent_count, cluster.get_all_title())

        return sentences, scores


if __name__ == '__main__':
    from src.loader.class_loader import Cluster, load_cluster

    SSOURCE = 'sent_splitted_token'

    dataset = load_cluster(
        "/home/hvn/Documents/dskt/vlsp-final-year/dataset/vlsp_2022_abmusu_train_data_new.jsonl",
        1,
    )
    print(dataset.clusters[0].documents[0])
    dataset.set_source(SSOURCE)

    config = load_config_from_json()
    mmr = MMRQuery(config.models[0])
    sents, rates = mmr.predict(dataset.clusters[0])

    print("** Predicted sent: \n", sents)
    print("** Rating: ", len(rates))
