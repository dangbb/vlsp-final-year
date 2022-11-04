import math
import random
from typing import List

from src.config.config import ModelConfig
from src.loader.class_loader import Dataset, Cluster


class RandomModel:
    def __init__(self, config: ModelConfig):
        super(RandomModel, self).__init__()

        self.config = config
        self.SENTENCES_COUNT = config.params

    def training(self, dataset: Dataset) -> None:
        pass

    def predict(self, cluster: Cluster) -> (List[str], List[float]):
        all_sents = cluster.get_all_sents()

        sent_count = len(all_sents)
        for SENT_COUNT in self.SENTENCES_COUNT:
            if 0 <= SENT_COUNT < 1:
                sent_count = min(int(math.ceil(len(all_sents) * SENT_COUNT)), sent_count)
            else:
                sent_count = min(int(SENT_COUNT), sent_count)

        chosen_idx = random.choices(range(0, len(all_sents)), k = sent_count)

        return [all_sents[idx] for idx in chosen_idx], [1 if i in chosen_idx else 0 for i in range(len(all_sents))]



