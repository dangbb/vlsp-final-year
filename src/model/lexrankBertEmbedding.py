import math
from collections import Counter, defaultdict
from typing import List

import numpy as np

from lexrank.algorithms.power_method import stationary_distribution
from lexrank.utils.text import tokenize

from src.config.config import ModelConfig, load_config_from_json
from src.loader.class_loader import Cluster
from src.model.model import Model
from src.utils.embedding import get_embedding
from src.utils.similarity import cosine_similarity


class LexRankBert:
    def __init__(self, config: ModelConfig, stopwords=None, keep_numbers=False, keep_emails=False,
                 keep_urls=False, include_new_words=True):
        if stopwords is None:
            self.stopwords = set()
        else:
            self.stopwords = stopwords

        self.keep_numbers = keep_numbers
        self.keep_emails = keep_emails
        self.keep_urls = keep_urls
        self.include_new_words = include_new_words

        self.config = config
        self.embedding = get_embedding(config.embedding)

    def get_summary(
        self,
        sentences,
        summary_size=1,
        threshold=.03,
        fast_power_method=True,
    ):
        if not isinstance(summary_size, int) or summary_size < 1:
            raise ValueError('\'summary_size\' should be a positive integer')

        lex_scores = self.rank_sentences(
            sentences,
            threshold=threshold,
            fast_power_method=fast_power_method,
        )

        sorted_ix = np.argsort(lex_scores)[::-1]
        summary = [sentences[i] for i in sorted_ix[:summary_size]]

        return summary

    def rank_sentences(
        self,
        sentences,
        threshold=.03,
        fast_power_method=True,
    ):
        if not (
            threshold is None or
            isinstance(threshold, float) and 0 <= threshold < 1
        ):
            raise ValueError(
                '\'threshold\' should be a floating-point number '
                'from the interval [0, 1) or None',
            )


        similarity_matrix = self._calculate_similarity_matrix(sentences)

        if threshold is None:
            markov_matrix = self._markov_matrix(similarity_matrix)

        else:
            markov_matrix = self._markov_matrix_discrete(
                similarity_matrix,
                threshold=threshold,
            )

        scores = stationary_distribution(
            markov_matrix,
            increase_power=fast_power_method,
            normalized=False,
        )

        return scores

    def tokenize_sentence(self, sentence):
        tokens = tokenize(
            sentence,
            self.stopwords,
            keep_numbers=self.keep_numbers,
            keep_emails=self.keep_emails,
            keep_urls=self.keep_urls,
        )

        return tokens

    def _calculate_idf(self, documents):
        bags_of_words = []

        for doc in documents:
            doc_words = set()

            for sentence in doc:
                words = self.tokenize_sentence(sentence)
                doc_words.update(words)

            if doc_words:
                bags_of_words.append(doc_words)

        if not bags_of_words:
            raise ValueError('documents are not informative')

        doc_number_total = len(bags_of_words)

        if self.include_new_words:
            default_value = math.log(doc_number_total + 1)

        else:
            default_value = 0

        idf_score = defaultdict(lambda: default_value)

        for word in set.union(*bags_of_words):
            doc_number_word = sum(1 for bag in bags_of_words if word in bag)
            idf_score[word] = math.log(doc_number_total / doc_number_word)

        return idf_score

    def _calculate_similarity_matrix(self, sentences: List[str]):

        length = len(sentences)
        similarity_matrix = np.zeros([length] * 2)

        embeddings = self.embedding.transform(sentences)

        for i in range(length):
            for j in range(i, length):
                similarity = cosine_similarity(embeddings[i], embeddings[j])

                if similarity:
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        return similarity_matrix

    def _markov_matrix(self, similarity_matrix):
        row_sum = similarity_matrix.sum(axis=1, keepdims=True)

        return similarity_matrix / row_sum

    def _markov_matrix_discrete(self, similarity_matrix, threshold):
        markov_matrix = np.zeros(similarity_matrix.shape)

        for i in range(len(similarity_matrix)):
            columns = np.where(similarity_matrix[i] > threshold)[0]
            markov_matrix[i, columns] = 1 / len(columns)

        return markov_matrix


class LexRankBertModel(Model):
    def __init__(self, config: ModelConfig):
        super(LexRankBertModel, self).__init__()

        self.config = config
        self.SENTENCES_COUNT = config.params

        self.model = LexRankBert(config)

    def predict(self, cluster: Cluster) -> (List[str], List[float]):
        all_sents = cluster.get_all_sents()

        sent_count = len(all_sents)
        for SENT_COUNT in self.SENTENCES_COUNT:
            if 0 <= SENT_COUNT < 1:
                sent_count = min(int(math.ceil(len(all_sents) * SENT_COUNT)), sent_count)
            else:
                sent_count = min(int(SENT_COUNT), sent_count)

        sentences = self.model.get_summary(all_sents, sent_count)
        scores = self.model.rank_sentences(all_sents, self.config.config["threshold"])

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
    lexrankBert = LexRankBertModel(config.models[0])
    sents, rates = lexrankBert.predict(dataset.clusters[0])

    print("** Predicted sent: \n", sents)
    print("** Rating: ", len(rates))
