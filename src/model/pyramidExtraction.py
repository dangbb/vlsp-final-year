from typing import List, Tuple, Any

from tqdm import tqdm

from src.config.config import ModelConfig, load_config_from_json
from src.loader.class_loader import Cluster
from src.model.model import Model
from underthesea import pos_tag
from rouge import Rouge

import sys
sys.setrecursionlimit(15000)

rouge = Rouge()


def get_entities_with_frequency(cluster: Cluster) -> List[str]:
    entities = {}

    for doc in cluster.documents:
        for sentence in doc.text_container.sent_splitted_token:
            for word, tag in pos_tag(sentence):
                if tag == 'Np':
                    entities[word] = entities.get(word, 0) + 1

    entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)
    return entities


def get_sentences_with_rouge(cluster: Cluster) -> List[Tuple[Any, int]]:
    sentences = []

    for doc in cluster.documents:
        sentences += doc.text_container.sent_splitted_token

    sentences_with_rouge = []

    raw_cluster_str = ''

    for doc in cluster.documents:
        for sentence in doc.text_container.sent_splitted_token:
            raw_cluster_str += sentence + ' '

    for sentence in sentences:
        score = 0
        try:
            rouge_score = rouge.get_scores(sentence, raw_cluster_str)
            score += rouge_score[0]['rouge-1']['f']
        except Exception as e:
            print(len(sentence), len(raw_cluster_str))
            pass
        sentences_with_rouge.append((sentence, score))

    sentences_with_rouge = sorted(sentences_with_rouge, key=lambda x: x[1], reverse=True)
    return sentences_with_rouge


class PyramidExt(Model):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    def predict(self, cluster: Cluster) -> (List[str], List[float]):
        entities = get_entities_with_frequency(cluster)
        sentences_with_rouge = get_sentences_with_rouge(cluster)
        num_sentences = len(sentences_with_rouge)

        if self.config.params[0] < 1:
            extracted_len = int(num_sentences * self.config.params[0])
        else:
            extracted_len = self.config.params[0]
        extracted_len = max(1, extracted_len)

        extracted_sentences = []
        scores = []

        for entity, freq in entities:
            for sentence, rouge_score in sentences_with_rouge:
                if entity in sentence:
                    extracted_sentences.append(sentence)
                    scores.append(rouge_score)
                    if len(extracted_sentences) >= extracted_len:
                        return extracted_sentences, scores
        return extracted_sentences, scores


if __name__ == '__main__':
    from src.loader.class_loader import load_cluster

    from src.loader.class_loader import Cluster, load_cluster

    SOURCE = 'sent_splitted_token'

    dataset = load_cluster(
        "/home/hvn/Documents/dskt/vlsp-final-year/dataset/vlsp_2022_abmusu_train_data_new.jsonl",
        50,
    )
    dataset.set_source(SOURCE)

    config = load_config_from_json()
    pyExt = PyramidExt(config.models[0])

    for cluster in tqdm(dataset.clusters):
        sent, rates = pyExt.predict(cluster)
        if len(rates) == 0:
            print(f"Debug at: {cluster.cluster_idx}")
