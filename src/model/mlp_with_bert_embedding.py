import math
from typing import List

from src.config.config import ModelConfig, load_config_from_json
from src.loader.class_loader import Cluster, Dataset
from src.model.model import Model
from src.utils.embedding import Embedding, get_embedding, SentenceCustomEmbedding
from src.utils.similarity import get_similarity

import torch
from torch import nn

from torch.utils.data import Dataset


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU(),
        )

        self.bert_embedder = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x[0] = bert vector embedding. (batch, 768)
        x[1] = lingustic vector embedding. (batch, 8)
        :param x:
        :return:
        """
        bert_embedding = self.bert_embedder(x[0])
        tensor = torch.Tensor(x)
        tensor = torch.cat((tensor, bert_embedding), 1)
        return self.layers(tensor)


class MLP_with_bert(Model):
    def __init__(self, config: ModelConfig):
        super(MLP_with_bert, self).__init__()
        self.config = config

        self.embedding: Embedding = get_embedding(config.embedding)
        self.lingustic_embedding = SentenceCustomEmbedding(load_config_from_json())

        self.model = MLP()

        self.threshold = 0.5
        self.is_train = False

    def training(self, dataset: Dataset):
        self.embedding.fit_dataset(dataset)
        self.lingustic_embedding.fit_dataset(dataset)

        """
        Training for MLP
        """
        self.model.load_state_dict(self.config.model_path)
        """
        End training for MLP
        """

        self.is_train = True
        pass

    def predict(self, cluster: Cluster) -> (List[str], List[float]):
        if not self.is_train:
            raise Exception("[MLP With BERT] Model not train yet")

        sentences = cluster.get_all_sents()
        bert_embeddings = self.embedding.fit(sentences)
        lingustic_embeddings = self.lingustic_embedding.fit(sentences)

        vector = []
        for i, sent in enumerate(sentences):
            vector.append(
                (
                    bert_embeddings[i],
                    lingustic_embeddings[i],
                )
            )

        self.model.eval()
        preds = self.model(vector)
        preds = preds.view(-1).data

        scores = []
        return_sentences = []
        for i, value in enumerate(preds):
            if value > self.threshold:
                sentences.append(sentences[i])
                scores.append(1.0)
            else:
                scores.append(0.0)

        return return_sentences, scores


if __name__ == '__main__':
    pass
