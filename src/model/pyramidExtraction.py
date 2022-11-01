from typing import List

from src.config.config import ModelConfig, load_config_from_json
from src.loader.class_loader import Cluster
from src.model.model import Model


class PyramidExt(Model):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    def predict(self, cluster: Cluster) -> (List[str], List[float]):
        pass
