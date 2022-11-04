import logging

import pandas as pd
from tqdm import tqdm

from src.config.config import Config
from src.loader.class_loader import Cluster, load_cluster, SOURCE
from src.utils.embedding import SentenceBertEmbedding
from src.utils.oracle import OracleConfig, OracleGenerator
from src.utils.similarity import cosine_similarity


class LabelingConfig:
    def __init__(self):
        self.candidate = [12, .2]
        self.chosen = [8, .1]
        self.threshold = 0.65
        self.pad_chosen = [15, .4]


class OracleLabeling:
    def __init__(self, config: LabelingConfig):
        logging.warning("[JOB {}] - start labeling config ...".format('ORACLE LABELING INIT'))
        self.config = config
        self.embedder = SentenceBertEmbedding(Config.load_config_from_json().embedding)

        logging.warning("[JOB {}] - labeling config init done".format('ORACLE LABELING INIT'))

    def __call__(self, cluster: Cluster):
        chosen_sent = len(cluster.get_all_sents())

        for size in self.config.chosen:
            if size < 1.0:
                chosen_sent = min(chosen_sent, len(cluster.get_all_sents()) * size)
            else:
                chosen_sent = min(chosen_sent, int(size))

        chosen_sent = max(2, int(chosen_sent))

        candidate_sent = len(cluster.get_all_sents())

        for size in self.config.candidate:
            if size < 1.0:
                candidate_sent = min(candidate_sent, len(cluster.get_all_sents()) * size)
            else:
                candidate_sent = min(candidate_sent, int(size))

        candidate_sent = max(2, int(candidate_sent))

        config = OracleConfig()
        config.candidate_sent = int(candidate_sent)
        config.chosen_sen = int(chosen_sent)

        oracle = OracleGenerator(config)

        com, _ = oracle.generate_oracle(cluster)

        new_cluster = cluster
        new_cluster.set_source(SOURCE.SENT_SPLITTED_TEXT.value)

        embeddings = self.embedder.transform(new_cluster.get_all_sents())

        label = []

        for idx, sent in enumerate(new_cluster.get_all_sents()):
            if idx not in com:
                for ref_idx in com:
                    if cosine_similarity(embeddings[idx], embeddings[ref_idx]) >= self.config.threshold:
                        label.append(1.0)
                        break
            else:
                label.append(1.0)
            if len(label) <= idx:
                label.append(0.0)

        assert len(label) == len(new_cluster.get_all_sents()), "label len {}, all sents len {}".format(len(label), len(new_cluster.get_all_sents()))

        return label



if __name__ == "__main__":
    config = LabelingConfig()
    model = OracleLabeling(config)

    # valid
    dataset = load_cluster(
        Config.load_config_from_json().valid_path,
    )
    dataset.set_source(SOURCE.SENT_SPLITTED_TOKEN.value)

    all_label = []
    for cluster in tqdm(dataset.clusters):
        all_label = all_label + model(cluster)

    df = pd.DataFrame()
    df["label"] = all_label

    df.to_csv("/home/dang/vlsp-final-year/dataset/embedding/oracle_label_valid.csv", index=False)
    # training
    dataset = load_cluster(
        Config.load_config_from_json().train_path,
    )
    dataset.set_source(SOURCE.SENT_SPLITTED_TOKEN.value)

    all_label = []
    for cluster in tqdm(dataset.clusters):
        all_label = all_label + model(cluster)

    df = pd.DataFrame()
    df["label"] = all_label

    df.to_csv("/home/dang/vlsp-final-year/dataset/embedding/oracle_label_train.csv", index=False)
