from typing import List

from src.loader.class_loader import Cluster, Dataset


class Model:
    def __init__(self):
        pass

    def training(self, dataset: Dataset) -> None:
        pass

    def predict(self, cluster: Cluster) -> (List[str], List[float]):
        pass
