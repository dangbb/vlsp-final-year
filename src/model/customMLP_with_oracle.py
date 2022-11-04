import logging
import math
import os
from typing import List

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.config.config import ModelConfig, Config
from src.loader.class_loader import Cluster, Dataset, SOURCE
from src.model.model import Model
from src.utils.embedding import SentenceCustomEmbedding
from src.utils.oracle_labeling import OracleLabeling, LabelingConfig


class DLConfig:
    def __init__(self):
        self.layers = (32, )
        self.max_iter = 500
        self.saved_path = "/home/dang/vlsp-final-year/dataset/embedding"


class CustomMLPWithOracle(Model):
    def __init__(self, config: ModelConfig):
        super(CustomMLPWithOracle, self).__init__()

        self.config = config

        self.custom_embedding = SentenceCustomEmbedding(Config.load_config_from_json())
        self.labeling = OracleLabeling(LabelingConfig())

        self.DLconfig = DLConfig()
        self.model = MLPClassifier(
            hidden_layer_sizes=self.DLconfig.layers,
            random_state=1,
            max_iter=self.DLconfig.max_iter
        )
        self.is_train = False

        self.SENTENCES_COUNT = config.params

    def training(self, dataset: Dataset) -> None:
        self.custom_embedding.fit_dataset(dataset)

        dataset.set_source(SOURCE.SENT_SPLITTED_TOKEN.value)
        dataset_embedding = self.custom_embedding.transform_dataset(dataset)

        dataset_label = []
        n_sent_all = 0

        logging.warning("[JOB {}] - get label...".format('custome mlp oracle'))

        for cluster in tqdm(dataset.clusters):
            dataset_label = dataset_label + self.labeling(
                cluster=cluster
            )
            n_sent_all += len(cluster.get_all_sents())

        dataset_embedding['label'] = dataset_label
        dataset_embedding.to_csv(
            os.path.join(self.DLconfig.saved_path, "oracle_label_custom_embedding.csv"),
            index=False,
        )

        logging.warning("[JOB {}] - done get label".format('custome mlp oracle'))

        print("training dataset shape: ", dataset_embedding.shape)
        train_ = dataset_embedding[[col for col in dataset_embedding.columns if col != 'label']]
        assert train_.shape[0] == len(dataset_label), \
            "feature and label size mismatch, {} vs {}, with num sentence {}".format(train_.shape, len(dataset_label), n_sent_all)
        self.model.fit(train_, pd.Series(dataset_label))
        self.is_train = True

    def predict(self, cluster: Cluster) -> (List[str], List[float]):
        if not self.is_train:
            logging.fatal('mlp model hasnt been trained')
            return [], []

        all_sents = cluster.get_all_sents()
        sent_count = len(all_sents)

        for SENT_COUNT in self.SENTENCES_COUNT:
            if 0 <= SENT_COUNT < 1:
                sent_count = min(int(math.ceil(len(all_sents) * SENT_COUNT)), sent_count)
            else:
                sent_count = min(int(SENT_COUNT), sent_count)

        data = self.custom_embedding.transform_cluster(cluster)
        feature_ = data[[col for col in data.columns if col != 'label']]

        pred_label = self.model.predict(feature_)

        chosen_sent_idx = np.argpartition(pred_label, -sent_count)[-sent_count:]
        chosen_sent = [all_sents[idx] for idx in chosen_sent_idx]

        return chosen_sent, pred_label

    def validate(self, valid_set: Dataset):
        logging.warning("[JOB {}] - start validate...".format('custome mlp oracle'))

        if not self.is_train:
            logging.fatal('mlp model hasnt been trained')
            return [], []

        self.custom_embedding.fit_dataset(valid_set)

        valid_set.set_source(SOURCE.SENT_SPLITTED_TOKEN.value)
        dataset_embedding = self.custom_embedding.transform_dataset(valid_set)

        dataset_label = []
        n_sent_all = 0

        logging.warning("[JOB {}] - get label...".format('custome mlp oracle'))

        for cluster in tqdm(valid_set.clusters):
            dataset_label = dataset_label + self.labeling(
                cluster=cluster
            )
            n_sent_all += len(cluster.get_all_sents())

        logging.warning("[JOB {}] - done get label".format('custome mlp oracle'))

        print("training dataset shape: ", dataset_embedding.shape)
        valid_ = dataset_embedding[[col for col in dataset_embedding.columns if col != 'label']]
        assert valid_.shape[0] == len(dataset_label), \
            "feature and label size mismatch, {} vs {}, with num sentence {}".format(valid_.shape, len(dataset_label),
                                                                                     n_sent_all)
        print("Validate result: ", self.model.score(valid_, pd.Series(dataset_label)))

        y_true = dataset_label
        y_pred = self.model.predict(valid_)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, normalize='all').ravel()
        print("TN: {} - FP: {} - FN: {} - TP: {}".format(tn, fp, fn, tp))

        logging.warning("[JOB {}] - validate done".format('custome mlp oracle'))


if __name__ == "__main__":
    from src.loader.class_loader import Cluster, load_cluster

    dataset = load_cluster(
        Config.load_config_from_json().train_path,
    )
    dataset.set_source(SOURCE.SENT_SPLITTED_TOKEN.value)

    valid_dataset = load_cluster(
        Config.load_config_from_json().valid_path,
    )
    valid_dataset.set_source(SOURCE.SENT_SPLITTED_TOKEN.value)

    n_sent = 0
    for cluster in dataset.clusters:
        n_sent += len(cluster.get_all_sents())
    print("Total sent: ", n_sent)

    config = Config.load_config_from_json()
    mlp = CustomMLPWithOracle(config.models[0])
    print("training")

    mlp.training(dataset)
    print("validate")

    mlp.validate(valid_dataset)
    print("predict")

    sents, rates = mlp.predict(dataset.clusters[0])

    print("** Predicted sent: \n", sents)
    print("** Rating: ", len(rates))