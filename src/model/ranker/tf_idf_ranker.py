import logging
from typing import List

from src.config.config import load_config_from_json
from src.loader.class_loader import Dataset, SOURCE

from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFRanker:
    def __init__(self):
        logging.warning("TFIDF-init: Start create a MMR Summarizer instance")

        self.cluster_model = TfidfVectorizer(max_df = .5)
        self.document_model = TfidfVectorizer(max_df = .5)
        self.sent_model = TfidfVectorizer(max_df = .5)
        self.is_train = False
        self.top_k = 10
        logging.warning("TFIDF-init: Model created")

    def training(self, dataset: Dataset):
        dataset.set_source(SOURCE.SENT_SPLITTED_TOKEN.value)

        cluster_sents = []
        document_sents = []

        for cluster in dataset.clusters:
            cluster_sents.append(' '.join(cluster.get_all_sents()))
            for doc in cluster.documents:
                document_sents.append(' '.join(doc.get_all_sents()))

        self.cluster_model.fit(cluster_sents)
        self.document_model.fit(document_sents)

        self.is_train = True

    def __call__(
            self,
            sentences: List[str],
    ) -> List[float]:
        if not self.is_train:
            raise Exception("Model hasn't trained yet")
        scores = []

        self.sent_model.fit(sentences)

        for sent in sentences:
            sent_embedding = self.sent_model.transform([sent]).toarray()[0]

            min_length = min(sent_embedding.shape[0], self.top_k)

            scores.append(sum(sorted(sent_embedding, reverse=True)[:min_length]) / min_length)

        return scores


if __name__ == '__main__':
    from src.loader.class_loader import load_cluster

    CSOURCE = 'sent_splitted_token'

    dataset = load_cluster(
        "/dataset/vlsp_abmusu_test_data.jsonl", 2
    )
    dataset.set_source(CSOURCE)

    config = load_config_from_json()
    tfidf = TFIDFRanker()
    tfidf.training(dataset)
    score = tfidf(dataset.clusters[0].get_all_sents())

    print("Score 0: ", score[0])
    print("Score 1: ", score[1])
    print("Score 2: ", score[2])



