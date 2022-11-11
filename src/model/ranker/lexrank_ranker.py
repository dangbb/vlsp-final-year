from typing import List

from lexrank import LexRank

from src.config.config import load_config_from_json
from src.loader.class_loader import Dataset, SOURCE
from src.model.model import Model


class LexrankRanker(Model):
    def __init__(self):
        super(LexrankRanker, self).__init__()

        self.cluster_model = None
        self.document_model = None
        self.sent_model = None

        self.is_train = False

    def training(self, dataset: Dataset) -> None:
        dataset.set_source(SOURCE.SENT_SPLITTED_TOKEN.value)

        cluster_sents = []
        document_sents = []

        for cluster in dataset.clusters:
            cluster_sents.append(' '.join(cluster.get_all_sents()))
            for doc in cluster.documents:
                document_sents.append(' '.join(doc.get_all_sents()))

        self.cluster_model = LexRank(cluster_sents)
        self.document_model = LexRank(document_sents)

        self.is_train = True
        pass

    def __call__(
            self,
            sentences: List[str]) -> List[float]:
        if not self.is_train:
            raise Exception("Model hasn't trained yet")
        self.sent_model = LexRank(sentences)

        sents = sentences

        return self.sent_model.rank_sentences(sents)


if __name__ == "__main__":
    from src.loader.class_loader import load_cluster

    dataset = load_cluster(
        "/home/hvn/Documents/dskt/vlsp-final-year/dataset/vlsp_2022_abmusu_train_data_new.jsonl",
        1,
    )

    config = load_config_from_json()
    lxr = LexrankRanker()
    lxr.training(dataset)
    score = lxr(dataset.clusters[0].get_all_sents())

    print("Score 0: ", score[0])
    print("Score 1: ", score[1])
    print("Score 2: ", score[2])
