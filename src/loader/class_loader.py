from tqdm import tqdm

import json
import string

from underthesea import word_tokenize, sent_tokenize
from typing import List

exclude = set(string.punctuation)
exclude.add('“')
exclude.add('”')
exclude.add('…')


def cleaning(sentences: List[str]):
    # remove dup
    new_sentences = list(dict.fromkeys(sentences))
    # remove empty sentence
    new_sentences = [sent for sent in new_sentences if sent.strip() != '']

    return new_sentences


class TextContainer:
    def __init__(self,
                 raw_text: List[str],
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

        sent_splitted_token: List[str] = []
        for sentence in sent_splitted_text:
            word_tokenized_sent = word_tokenize(sentence, format="text")
            word_tokenized_sent = word_tokenized_sent.split(' ')
            word_tokenized_sent = [word for word in word_tokenized_sent if word not in exclude]
            sent_splitted_token.append(' '.join(word_tokenized_sent))

        self.tokenized_text = cleaning(tokenized_text)
        self.sent_splitted_text = cleaning(sent_splitted_text)
        self.sent_splitted_token = cleaning(sent_splitted_token)


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


def load_cluster(path: str, n_cluster: int = -1) -> Dataset:
    dataset = Dataset()

    with open(path, 'r') as json_file:
        json_list = list(json_file)

        print("Total number of cluster: ", len(json_list))

        # dict_keys(['single_documents', 'summary', 'category'])
        # dict_keys(['title', 'anchor_text', 'raw_text'])

        for cluster_id, json_str in tqdm(enumerate(json_list)):

            result = json.loads(json_str)

            summary = ''
            try:
                summary = result['summary']
            except KeyError:
                pass

            cluster = Cluster(
                cluster_idx=cluster_id + 1,
                summary=summary,
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
