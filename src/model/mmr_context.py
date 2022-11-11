import logging
import math
from typing import List

import numpy as np

from src.config.config import ModelConfig, load_config_from_json
from src.evaluate.rouge_evaluator import PipRouge
from src.loader.class_loader import Cluster, SOURCE
from src.model.model import Model
from src.utils.embedding import Embedding, get_embedding
from src.utils.similarity import get_similarity


class MMRSummarizer2Context:
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
            anchor: List[str]
    ):
        self.embedding.fit(sentences)

        chosen_sentences_set = []
        chosen_com_set = []
        unchosen_sentences = [i for i in range(len(sentences) - 1)]

        combination_sentences_generator = [sentences[i] + ' ' + sentences[i + 1] for i in range(len(sentences) - 1)]

        embedding_vector = self.embedding.transform(combination_sentences_generator)
        embedding_document = self.embedding.transform(['.'.join(sentences)])[0]

        sigma = self.config.sigma
        scores = np.zeros(len(sentences))

        while len(chosen_sentences_set) < n_sent:
            best_score = -1.0
            best_sent_idx = -1
            for i in unchosen_sentences:
                salient = self.similarity(
                    embedding_vector[i],
                    embedding_document,
                )
                if len(chosen_sentences_set) == 0:
                    redundancy = 0
                else:
                    redundancy = max([
                        self.similarity(
                            embedding_vector[i],
                            embedding_vector[j]
                        ) for j in chosen_com_set
                    ])
                score = salient * sigma - redundancy * (1 - sigma)
                if score > best_score:
                    best_score = score
                    best_sent_idx = i

            if best_sent_idx == -1:
                break
            if best_sent_idx not in chosen_sentences_set:
                chosen_sentences_set.append(best_sent_idx)
            if best_sent_idx + 1 not in chosen_sentences_set:
                chosen_sentences_set.append(best_sent_idx + 1)
            chosen_com_set.append(best_sent_idx)

            unchosen_sentences.remove(best_sent_idx)

        for index in chosen_sentences_set:
            scores[index] = 1.0

        return [sentences[i] for i in chosen_sentences_set], scores.tolist()


class MMRQueryAnchorContext(Model):
    def __init__(self, config: ModelConfig):
        super(MMRQueryAnchorContext, self).__init__()
        self.embedding: Embedding = get_embedding(config.embedding)
        self.similarity = get_similarity(config.embedding)
        self.summarizer = MMRSummarizer2Context(config)

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

        rouge = PipRouge()
        best_score = -1
        best_title = ""
        for title in cluster.get_all_anchor():
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

    SSOURCE = 'sent_splitted_token'

    dataset = load_cluster(
        "/home/hvn/Documents/dskt/vlsp-final-year/dataset/vlsp_abmusu_test_data.jsonl",
        start=64, end=66,
    )
    dataset.set_source(SSOURCE)

    config = load_config_from_json()
    mmr = MMRQueryAnchorContext(config.models[0])
    sents, rates = mmr.predict(dataset.clusters[0])
    sents, rates = mmr.predict(dataset.clusters[1])
    sents, rates = mmr.predict(dataset.clusters[2])

    print("** Predicted sent: \n", sents)
    print("** Rating: ", len(rates))
