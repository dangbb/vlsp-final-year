import json

from underthesea import word_tokenize, sent_tokenize
from typing import List
from tqdm import tqdm


class Document:
    def __init__(self, cluster_idx: int, idx: int, title: str, anchor_text: str, raw_text: str):
        self.cluster_idx: int = cluster_idx
        self.idx: int = idx
        self.title: str = title
        self.anchor_text: List[str] = anchor_text.split('\n')
        self.raw_text: List[str] = raw_text.split('\n')

        self.preprocessed_text: List[str] = []  # preprocessed sentences (convert ellipsis)
        self.tokenized_text: List[str] = []  # default sentences, word tokenized
        self.sent_splitted_text: List[str] = []  # sentences tokenized
        self.sent_splitted_token: List[str] = []  # sentences tokenized, word tokenized

        tokenized_text: List[str] = []
        for sentence in self.raw_text:
            tokenized_text.append(word_tokenize(sentence, format="text"))

        sent_splitted_text: List[str] = []
        for sentence in self.raw_text:
            sent_splitted_text = sent_splitted_text + sent_tokenize(sentence)

        sent_splitted_token: List[str] = []
        for sentence in sent_splitted_text:
            sent_splitted_token.append(word_tokenize(sentence, format="text"))

        self.tokenized_text = tokenized_text
        self.sent_splitted_text = sent_splitted_text
        self.sent_splitted_token = sent_splitted_token

    def __str__(self):
        return f"Document {self.idx} - Cluster {self.cluster_idx}\n" + f"Title: {self.title}\n" + f"Anchor Text: {self.anchor_text}\n" + f"Raw Text:\n{self.raw_text}\n"

    def tokenize(self, tokenizer):
        self.tokenized_text = tokenizer(self.raw_text)


class Cluster:
    def __init__(self, cluster_idx: int, summary: List[str]):
        self.documents: List[Document] = []
        self.summary: List[str] = summary
        self.cluster_idx = cluster_idx

    def __len__(self):
        return len(self.documents)

    def add(self, doc: Document) -> None:
        self.documents.append(doc)


class Dataset:
    def __init__(self):
        self.clusters: List[Cluster] = []

    def add(self, cluster: Cluster):
        self.clusters.append(cluster)


def load_cluster(path: str) -> Dataset:
    dataset = Dataset()

    with open(path, 'r') as json_file:
        json_list = list(json_file)

        print("Total number of cluster: ", len(json_list))

        # dict_keys(['single_documents', 'summary', 'category'])
        # dict_keys(['title', 'anchor_text', 'raw_text'])

        for cluster_id, json_str in tqdm(enumerate(json_list)):

            result = json.loads(json_str)

            cluster = Cluster(cluster_idx=cluster_id + 1, summary=result['summary'])

            for document_id, single_doc in enumerate(result['single_documents']):
                cluster.add(Document(
                    cluster_idx=cluster_id + 1,
                    idx=document_id + 1,
                    title=single_doc['title'],
                    anchor_text=single_doc['anchor_text'],
                    raw_text=single_doc['raw_text']
                ))

            dataset.add(cluster)
    return dataset