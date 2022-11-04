import logging

from tqdm import tqdm

import json
import string

from underthesea import word_tokenize, sent_tokenize
from typing import List

from src.config.config import Config

exclude = set(string.punctuation)
exclude.add('“')
exclude.add('”')
exclude.add('…')

from enum import Enum


class SOURCE(Enum):
    RAW_TEXT = 'raw_text'
    TOKENIZED_TEXT = 'tokenized_text'
    SENT_SPLITTED_TEXT = 'sent_splitted_text'
    SENT_SPLITTED_TOKEN = 'sent_splitted_token'


def is_punc(sentence: str):
    for char in sentence:
        if char not in exclude and char != ' ':
            return False
    return True


def len_no_punc(sentence: str):
    count = 0
    for token in sentence:
        if token in exclude and token != '_':
            continue
        count += 1
    return count


def cleaning(sentences: List[str]):
    new_sentences = []

    for sent in sentences:
        new_sentences.append(''.join([char for char in sent if char not in exclude or char == '_']))

    # remove empty sentence and sentence contain only punc
    new_sentences = [
        sent.strip() for sent in new_sentences if sent.strip() != '' and
                                          sent.strip() not in exclude and
                                          not is_punc(sent.strip()) and
                                          len_no_punc(sent.strip()) > 10
    ]
    # remove dup
    new_sentences = list(dict.fromkeys(new_sentences))

    return new_sentences


class TextContainer:
    def __init__(self,
                 raw_text: str,
                 ):

        self.raw_text = raw_text.split('\n')

        tokenized_text: List[str] = []
        for sentence in self.raw_text:
            word_tokenized_sent = word_tokenize(sentence, format="text")
            word_tokenized_sent = word_tokenized_sent.split(' ')
            word_tokenized_sent = [word for word in word_tokenized_sent if word not in exclude]
            tokenized_text.append(' '.join(word_tokenized_sent))

        sent_splitted_text: List[str] = []
        for sentence in self.raw_text:
            sent_splitted_text = sent_splitted_text + sent_tokenize(sentence)
        self.sent_splitted_text = cleaning(sent_splitted_text)

        sent_splitted_token: List[str] = []
        for sentence in self.sent_splitted_text:
            word_tokenized_sent = word_tokenize(sentence, format="text")
            word_tokenized_sent = word_tokenized_sent.split(' ')
            word_tokenized_sent = [word for word in word_tokenized_sent if word not in exclude]
            sent_splitted_token.append(' '.join(word_tokenized_sent))

        self.sent_splitted_token = cleaning(sent_splitted_token)

        if len(self.sent_splitted_token) != len(self.sent_splitted_text):
            print("Token:\n", self.sent_splitted_token)
            print("Text:\n", self.sent_splitted_text)


class Document:
    def __init__(
            self,
            cluster_idx: int,
            idx: int,
            title: str,
            anchor_text: str,
            raw_text: str,
    ):
        self.cluster_idx: int = cluster_idx
        self.idx: int = idx
        self.title: str = title
        self.anchor_text: str = anchor_text

        self.text_container: TextContainer = TextContainer(raw_text)

    def __str__(self):
        return f"Document {self.idx} - Cluster {self.cluster_idx}\n" + f"Title: {self.title}\n" + f"Anchor Text: {self.anchor_text}\n" + f"Raw Text:\n{self.text_container.raw_text}\n"

    def tokenize(self, tokenizer):
        self.tokenized_text = tokenizer(self.raw_text)


class Cluster:
    def __init__(
            self,
            cluster_idx: int,
            summary: List[str],
            category: str,
    ):
        self.documents: List[Document] = []
        self.summary: TextContainer = TextContainer(summary)

        self.cluster_idx = cluster_idx
        self.category = category
        self.source = 'raw_text'

    def __len__(self):
        return len(self.documents)

    def set_source(self, new_source: str):
        if not new_source in ['raw_text', 'tokenized_text', 'sent_splitted_text', 'sent_splitted_token']:
            print("Unsupported source")
        self.source = new_source

    def add(self, doc: Document) -> None:
        self.documents.append(doc)

    def get_all_sents(self) -> List[str]:
        sents: List[str] = []

        for doc in self.documents:
            for sent in doc.text_container.__getattribute__(self.source):
                if sent not in sents:
                    sents.append(sent)

        return sents

    def get_summary(self) -> List[str]:
        return self.summary.__getattribute__(self.source)

    def get_all_title(self) -> List[str]:
        titles = []
        for doc in self.documents:
            titles.append(doc.title)
        return titles

    def get_all_anchor(self) -> List[str]:
        anchors = []
        for doc in self.documents:
            anchors.append(doc.anchor_text)
        return anchors


class Dataset:
    def __init__(self):
        self.clusters: List[Cluster] = []
        self.source = 'raw_text'

    def set_source(self, new_source: str):
        if not new_source in ['raw_text', 'preprocessed_text', 'tokenized_text', 'sent_splitted_text',
                              'sent_splitted_token']:
            print("Unsupported source")
        self.source = new_source

        for i in range(len(self.clusters)):
            self.clusters[i].set_source(new_source)

    def add(self, cluster: Cluster):
        self.clusters.append(cluster)


def load_cluster(path: str, n_cluster: int = -1, start: int = -1, end: int = -1) -> Dataset:
    dataset = Dataset()

    with open(path, 'r') as json_file:
        json_list = list(json_file)

        print("Total number of cluster: ", len(json_list))

        # dict_keys(['single_documents', 'summary', 'category'])
        # dict_keys(['title', 'anchor_text', 'raw_text'])

        for cluster_id, json_str in tqdm(enumerate(json_list)):
            if start != -1:
                if cluster_id < start:
                    continue

            if end != -1:
                if cluster_id > end:
                    break
            result = json.loads(json_str)

            if 'summary' in result.keys():
                cluster = Cluster(
                    cluster_idx=cluster_id + 1,
                    summary=result['summary'],
                    category=result['category'],
                )
            else:
                cluster = Cluster(
                    cluster_idx=cluster_id + 1,
                    summary="",
                    category=result['category'],
                )

            for document_id, single_doc in enumerate(result['single_documents']):
                cluster.add(Document(
                    cluster_idx=cluster_id + 1,
                    idx=document_id + 1,
                    title=single_doc['title'],
                    anchor_text=single_doc['anchor_text'],
                    raw_text=single_doc['raw_text']
                ))

            dataset.add(cluster)

            if cluster_id == n_cluster:
                break
    return dataset


if __name__ == "__main__":
    dataset = load_cluster(
        Config.load_config_from_json().train_path, 2
    )

    min_len = 1000
    min_sent = ""

    total_a = 0
    total_b = 0

    for cluster in dataset.clusters:
        dataset.set_source(SOURCE.SENT_SPLITTED_TEXT.value)
        a = len(cluster.get_all_sents())
        sent_a = cluster.get_all_sents()

        dataset.set_source(SOURCE.SENT_SPLITTED_TOKEN.value)
        b = len(cluster.get_all_sents())
        sent_b = cluster.get_all_sents()

        total_a += a
        total_b += b

        for sent in cluster.get_all_sents():
            if len(sent) < min_len:
                min_len = len(sent)
                min_sent = sent

        if a != b:
            print("Cluster {} - inconsistent".format(cluster.cluster_idx))

            for i in range(min(a, b)):
                print("i: ", i)
                print("A:\t", sent_a[i])
                print("B:\t", sent_b[i])
                print()

    print("Minimum sentence len:\n", min_sent)
    print("Total sen A: ", total_a)
    print("Total sen B: ", total_b)

    print("All titles: ", dataset.clusters[0].get_all_title())
