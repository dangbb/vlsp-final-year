import logging
import math
from typing import List

import numpy as np
from sklearn.neural_network import MLPRegressor

from src.config.config import ModelConfig, Config
from src.loader.class_loader import Cluster, Dataset
from src.model.model import Model
from src.utils.embedding import SentenceCustomEmbedding


class CustomMLP(Model):
    def __init__(self, config: ModelConfig):
        super(CustomMLP, self).__init__()

        self.config = config

        self.custom_embedding = SentenceCustomEmbedding(Config.load_config_from_json())
        self.model = MLPRegressor(hidden_layer_sizes=(32, 16), random_state=1, max_iter=500)
        self.is_train = False

        self.SENTENCES_COUNT = config.params

    def training(self, dataset: Dataset) -> None:
        dataset.set_source('sent_splitted_token')

        self.custom_embedding.fit_dataset(dataset)

        dataset_embedding = self.custom_embedding.transform_dataset(dataset)
        train_ = dataset_embedding[[col for col in dataset_embedding.columns if col != 'label']]
        label_ = dataset_embedding['label']

        self.model.fit(train_, label_)
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

if __name__ == "__main__":
    from src.loader.class_loader import Cluster, load_cluster

    SOURCE = 'sent_splitted_token'

    dataset = load_cluster(
        "/home/hvn/Documents/dskt/vlsp-final-year/dataset/vlsp_2022_abmusu_train_data_new.jsonl",
        1,
    )
    dataset.set_source(SOURCE)

    config = Config.load_config_from_json()
    mlp = CustomMLP(config.models[0])
    mlp.training(dataset)
    sents, rates = mlp.predict(dataset.clusters[0])

    print("** Predicted sent: \n", sents)
    print("** Rating: ", len(rates))