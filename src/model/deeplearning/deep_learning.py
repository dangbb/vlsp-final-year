import logging
import os
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from underthesea import ner, pos_tag

from src.config.config import EmbeddingConfig, Config
from src.loader.class_loader import Dataset, SOURCE
from src.model.deeplearning.deep_learning_label import get_label
from src.utils.embedding import SentenceBertEmbedding

MAX_LENGTH = 32
BERT_LENGTH = 768
TFIDF_LENGTH = 100


DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()
DEFAULT_EMBEDDING_CONFIG.bart_path = "/home/dang/vlsp-final-year/external/sentence_transformer/vn_sbert_deploy/phobert_base_mean_tokens_NLI_STS"


class CustomDLConfig:
    def __init__(self):
        self.layers = []


def extract_sent_data(sentence: str, bert_embedding, tfidf_embediding) -> List[float]:
    ner_extracted = ner(sentence)

    ner_start_embedding = [1.0 if ner_extracted[i][3].startswith('B') else 0.0 for i in range(len(ner_extracted))]
    ner_posit_embedding = [1.0 if ner_extracted[i][3].startswith('B') or ner_extracted[i][3].startswith('I') else 0.0 for i in range(len(ner_extracted))]

    pos_extracted = pos_tag(sentence)

    pos_noun_extracted = [1.0 if pos_extracted[i][1] == 'N' else 0.0 for i in range(len(pos_extracted))]
    pos_verb_extracted = [1.0 if pos_extracted[i][1] == 'V' else 0.0 for i in range(len(pos_extracted))]
    pos_adj_extracted = [1.0 if pos_extracted[i][1] == 'D' else 0.0 for i in range(len(pos_extracted))]

    while len(ner_start_embedding) < MAX_LENGTH:
        ner_start_embedding.append(0.0)
        ner_posit_embedding.append(0.0)
    while len(pos_noun_extracted) < MAX_LENGTH:
        pos_noun_extracted.append(0.0)
        pos_verb_extracted.append(0.0)
        pos_adj_extracted.append(0.0)

    return list(tfidf_embediding) + pos_noun_extracted + pos_verb_extracted + pos_adj_extracted + ner_start_embedding + ner_posit_embedding + list(bert_embedding)


def dataset_crafter(path: str, filename: str, dataset: Dataset, label_list: List[float]):
    dataset.set_source(SOURCE.SENT_SPLITTED_TEXT.value)

    sent_count = 0
    for cluster in dataset.clusters:
        sent_count = sent_count + len(cluster.get_all_sents())

    if label_list is not None:
        assert sent_count == len(label_list), "Mismatch number of feature and label, {} vs {}".format(sent_count, len(label_list))

    bert_embedder = SentenceBertEmbedding(DEFAULT_EMBEDDING_CONFIG)
    tfidf_embedder = TfidfVectorizer(max_df=.5, max_features=TFIDF_LENGTH)

    all_sents_text = []
    all_sents_token = []

    dataset.set_source(SOURCE.SENT_SPLITTED_TEXT.value)
    for cluster in dataset.clusters:
        all_sents_text = all_sents_text + cluster.get_all_sents()

    dataset.set_source(SOURCE.SENT_SPLITTED_TOKEN.value)
    for cluster in dataset.clusters:
        all_sents_token = all_sents_token + cluster.get_all_sents()

    bert_embedding = bert_embedder.transform(all_sents_text)
    cursor = 0
    tfidf_embedding = []

    for i, cluster in tqdm(enumerate(dataset.clusters), "Get tfidf embedding"):
        tfidf_embedder.fit(cluster.get_all_sents())
        tfidf_embedding = tfidf_embedding + list(tfidf_embedder.transform(cluster.get_all_sents()).toarray())

    assert len(bert_embedding) == len(tfidf_embedding)

    new_bert_embedding = []
    new_tfidf_embedding = []
    new_label = []
    sents = []

    for cluster in tqdm(dataset.clusters, "Concat bert and tf-idf embedding"):
        for sent in cluster.get_all_sents():
            if len(sent.split(' ')) < MAX_LENGTH:
                new_bert_embedding.append(bert_embedding[cursor])
                new_tfidf_embedding.append(tfidf_embedding[cursor])
                if label_list is not None:
                    new_label.append(label_list[cursor])
                else:
                    new_label.append(0.0)
                sents.append(sent)
            cursor += 1

    data = []
    for i in tqdm(range(len(sents)), "Make dataset"):
        data.append(extract_sent_data(sents[i], bert_embedding[i], tfidf_embedding[i]))

    np.savetxt(os.path.join(path, filename + '_feature.csv'), data, delimiter=',')
    np.savetxt(os.path.join(path, filename + '_label.csv'), new_label, delimiter=',')
    logging.warning("[JOB - Transform data] Save data done hmu hmu")


def data_loader(path: str, filename: str):
    data = np.loadtxt(os.path.join(path, filename), dtype=float, delimiter=',')
    return data


if __name__ == "__main__":
    from src.loader.class_loader import load_cluster

    train_set = load_cluster(
        Config.load_config_from_json().train_path,
    )
    train_set.set_source(SOURCE.SENT_SPLITTED_TOKEN.value)

    valid_set = load_cluster(
        Config.load_config_from_json().valid_path,
    )
    valid_set.set_source(SOURCE.SENT_SPLITTED_TOKEN.value)

    test_set = load_cluster(
        "/home/dang/vlsp-final-year/dataset/vlsp_abmusu_test_data.jsonl",
    )
    test_set.set_source(SOURCE.SENT_SPLITTED_TOKEN.value)

    train_label = []
    valid_label = []

    for cluster in train_set.clusters:
        label = get_label(cluster)
        print(len(label))
        train_label = train_label + label

    for cluster in valid_set.clusters:
        label = get_label(cluster)
        valid_label = valid_label + label

    dataset_crafter('/home/dang/vlsp-final-year/dataset/embedding', 'train', train_set, train_label)
    dataset_crafter('/home/dang/vlsp-final-year/dataset/embedding', 'train', valid_set, valid_label)
    dataset_crafter('/home/dang/vlsp-final-year/dataset/embedding', 'train', train_set, None)

    train_feature = data_loader(
        '/home/dang/vlsp-final-year/dataset/embedding',
        'train_feature.csv'
    )
    train_label = data_loader(
        '/home/dang/vlsp-final-year/dataset/embedding',
        'train_label.csv'
    )
    print(train_feature.shape)
    print(train_label.shape)

