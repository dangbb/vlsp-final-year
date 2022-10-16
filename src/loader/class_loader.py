import json
from nltk import pr

from underthesea import word_tokenize, sent_tokenize, pos_tag
from typing import List
from tqdm import tqdm
from rouge import Rouge


rouge = Rouge()


class Document:
    def __init__(self, cluster_idx: int, idx: int, title: str, anchor_text: str, raw_text: str):
        self.cluster_idx: int = cluster_idx
        self.idx: int = idx
        self.title: str = title
        self.anchor_text: str = anchor_text
        self.raw_text: str = raw_text
        self.raw_sentences: List[str] = raw_text.split('\n')

        self.preprocessed_text: List[str] = []  # preprocessed sentences (convert ellipsis)
        self.tokenized_text: List[str] = []  # default sentences, word tokenized
        self.sent_splitted_text: List[str] = []  # sentences tokenized
        self.sent_splitted_token: List[str] = []  # sentences tokenized, word tokenized

        tokenized_text: List[str] = []
        for sentence in self.raw_sentences:
            tokenized_text.append(word_tokenize(sentence, format="text"))

        sent_splitted_text: List[str] = []
        for sentence in self.raw_sentences:
            sent_splitted_text = sent_splitted_text + sent_tokenize(sentence)

        sent_splitted_token: List[str] = []
        for sentence in sent_splitted_text:
            sent_splitted_token.append(word_tokenize(sentence, format="text"))

        self.tokenized_text = tokenized_text
        self.sent_splitted_text = sent_splitted_text
        self.sent_splitted_token = sent_splitted_token

    def __str__(self):
        return f"Document {self.idx} - Cluster {self.cluster_idx}\n" + f"Title: {self.title}\n" + f"Anchor Text: {self.anchor_text}\n" + f"Raw Sentences:\n{self.raw_sentences}\n"

    def tokenize(self, tokenizer):
        self.tokenized_text = tokenizer(self.raw_sentences)


class Cluster:
    def __init__(self, cluster_idx: int, summary: List[str]):
        self.documents: List[Document] = []
        self.summary: str = summary
        self.cluster_idx = cluster_idx
        self.entities = None
        self.sentences_with_rouge = None

        doc_len = 0
        for doc in self.documents:
            doc_len += len(doc.raw_text)

        self.compressed_ratio: float = len(summary) / doc_len

    def __len__(self):
        return len(self.documents)

    def add(self, doc: Document) -> None:
        self.documents.append(doc)
    
    def get_entities_with_frequencies(self):
        if self.entities is not None:
            return self.entities

        self.entities = {}
        for doc in self.documents:
            for sentence in doc.sent_splitted_token:
                for word, tag in pos_tag(sentence):
                    if tag == 'Np':
                        self.entities[word] = self.entities.get(word, 0) + 1
        
        # sort entities by frequencies
        self.entities = sorted(self.entities.items(), key=lambda x: x[1], reverse=True)
        return self.entities
    
    def get_sentences_with_rouge(self):
        if self.sentences_with_rouge is not None:
            return self.sentences_with_rouge

        sentences = []
        for doc in self.documents:
            sentences += doc.sent_splitted_text
        
        self.sentences_with_rouge = []
        for sentence in sentences:
            score = 0
            for doc in self.documents:
                if sentence in doc.sent_splitted_text:
                    continue
                rouge_score = rouge.get_scores(sentence, doc.raw_text)
                score += rouge_score[0]['rouge-1']['f']
            self.sentences_with_rouge.append((sentence, score))
        
        # sort sentences by rouge score
        self.sentences_with_rouge = sorted(self.sentences_with_rouge, key=lambda x: x[1], reverse=True)

        return self.sentences_with_rouge
        

    def get_pyramid_based_masked_sentences(self, masked_ratio=0.1):
        entities = self.get_entities_with_frequencies()
        sentences_with_rouge = self.get_sentences_with_rouge()
        num_sentences = len(sentences_with_rouge)
        num_masked_sentences = int(num_sentences * masked_ratio)

        masked_sentences = []
        
        for entity, freq in entities:
            if freq < 2:
                break
            for sentence, rouge_score in sentences_with_rouge:
                if entity in sentence:
                    print(sentence)
                    masked_sentences.append(sentence)
                    if len(masked_sentences) >= num_masked_sentences:
                        return masked_sentences
        
        return masked_sentences


class Dataset:
    def __init__(self):
        self.clusters: List[Cluster] = []

    def add(self, cluster: Cluster):
        self.clusters.append(cluster)


def load_cluster(path: str) -> Dataset:
    dataset = Dataset()

    with open(path, 'r', encoding='utf-8') as json_file:
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
            doc_len = 0
            for doc in cluster.documents:
                doc_len += len(doc.raw_text)
            cluster.compressed_ratio = len(cluster.summary) / doc_len
            dataset.add(cluster)
    return dataset
