from tqdm import tqdm

import os
import pandas as pd

from underthesea import word_tokenize, sent_tokenize

from src.config.config import load_config_from_json
from src.loader.class_loader import Dataset, load_cluster

STATISTIC_FOLDER_NAME = 'stat'


def calc_stats(dataset: Dataset, saved_path: str):
    # Attribute for each cluster
    cluster_statistic = pd.DataFrame(columns=[
        'cluster_idx',
        'n_doc',  # Number of documents
        'n_sent',  # Number of sentences
        'n_word_summary',  # word in summary
        'n_sent_summary',  # sent in summary
        'avg_word_per_sent_summary'  # Avg word per sent, equals to total number of words / total number of sent
    ])

    for cluster in tqdm(dataset.clusters):
        n_doc = len(cluster.documents)

        splitted_summary = sent_tokenize(cluster.summary.raw_str)
        n_sent_summary = len(splitted_summary)
        n_word_summary = sum([len(word_tokenize(sent)) for sent in splitted_summary])

        n_sent = 0
        for doc in cluster.documents:
            n_sent = n_sent + len(doc.text_container.raw_text)

        cluster_statistic = cluster_statistic.append({
            'cluster_idx': cluster.cluster_idx,
            'n_doc': n_doc,
            'n_sent': n_sent,
            # 'n_word_summary': n_word_summary,
            # 'n_sent_summary': n_sent_summary,
            # 'avg_word_per_sent_summary': n_word_summary / n_sent_summary,
        }, ignore_index=True)

    cluster_statistic.to_csv(os.path.join(saved_path,
                                          os.path.join(
                                              STATISTIC_FOLDER_NAME,
                                              'cluster_test.csv'
                                          )), index=False)


def calc_doc_stats(dataset: Dataset, saved_path: str):
    # Attribute for each document
    document_statistic = pd.DataFrame(columns=[
        'cluster_idx',
        'document_idx',
        'n_sent',  # number of sentence in document
        'n_word',  # number of word in document
        'avg_word_per_sent',  # number of word per sentence
        'title_len',  # Number of word in title
        'anchor_len',  # Number of word in anchor
    ])

    for cluster in tqdm(dataset.clusters):
        for doc in cluster.documents:
            n_sent = len(doc.raw_text)
            n_word = sum([len(word_tokenize(sent)) for sent in doc.raw_text])
            title_len = len(word_tokenize(doc.title))
            anchor_len = len(word_tokenize(doc.anchor_text))

            document_statistic = document_statistic.append({
                'cluster_idx': doc.cluster_idx,
                'document_idx': doc.idx,
                'n_sent': n_sent,
                'n_word': n_word,
                'avg_word_per_sent': n_word / n_sent,
                'title_len': title_len,
                'anchor_len': anchor_len,
            }, ignore_index=True)

    document_statistic.to_csv(os.path.join(saved_path,
                                           os.path.join(
                                               STATISTIC_FOLDER_NAME,
                                               'document.csv'
                                           )), index=False)


def get_statistic_record(df: pd.DataFrame, saved_path: str, filename: str):
    statistic = pd.DataFrame(columns=[
        'colname',
        'mean',
        'max',
        'min',
        'std',
    ])
    for col in df.columns:
        describe = df[col].describe()
        statistic = statistic.append({
            'colname': col,
            'mean': describe['mean'],
            'max': describe['max'],
            'min': describe['min'],
            'std': describe['std']
        }, ignore_index=True)

    statistic.to_csv(os.path.join(
        saved_path,
        os.path.join(
            STATISTIC_FOLDER_NAME,
            filename
        )
    ), index=False)
    return statistic


# config = load_config_from_json()
# dataset = load_cluster(
#     "/home/hvn/Documents/dskt/vlsp-final-year/dataset/vlsp_abmusu_test_data.jsonl",
# )
#
# save_path = "/home/hvn/Documents/dskt/vlsp-final-year/data/statistic"
#
# calc_stats(dataset, save_path)

df = pd.read_csv("/home/hvn/Documents/dskt/vlsp-final-year/data/statistic/stat/cluster_test.csv")
get_statistic_record(df, "/home/hvn/Documents/dskt/vlsp-final-year/data/statistic", 'cluster_test_stats.csv')
