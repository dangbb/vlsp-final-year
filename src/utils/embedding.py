import logging
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer

from external.sentence_transformer.sentence_transformers.sentence_transformers import SentenceTransformer
from src.config.config import EmbeddingConfig, load_config_from_json


class Embedding:
    def __init__(self):
        pass

    def fit(self, sentences: List[str]):
        pass

    def transform(self, sentences: List[str]):
        pass


class TfIdfEmbedding(Embedding):
    def __init__(self, config: EmbeddingConfig, stopword: List[str] = None):
        super().__init__()
        logging.warning('Start create TF-IDF embedding')

        self.config = config
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words=stopword,
            max_df=self.config.max_df
            )
        logging.warning('Create TF-IDF embedding complete')

    def fit(self, sentences: List[str]):
        self.tfidf_vectorizer.fit(sentences)

    def transform(self, sentences: List[str]):
        return self.tfidf_vectorizer.transform(sentences).toarray()


class SentenceBertEmbedding(Embedding):
    def __init__(self, config: EmbeddingConfig):
        logging.warning('Start create SBERT embedding')
        super().__init__()
        self.config = config
        self.model = SentenceTransformer(
            self.config.bart_path
        )
        logging.warning('Create SBERT embedding complete')

    def fit(self, sentences: List[str]):
        pass

    def transform(self, sentences: List[str]):
        return self.model.encode(sentences=sentences)


def get_embedding(config: EmbeddingConfig) -> Embedding:
    if config.model == 'SBERT':
        return SentenceBertEmbedding(config)
    elif config.model == 'tf-idf':
        return TfIdfEmbedding(config)
    else:
        raise Exception('Unsupported embedding {}'.format(config.model))


if __name__ == "__main__":
    sample_text = ["John like horror movie", "Ryan watches dramatic movies"]
    config = load_config_from_json()
    embedding = SentenceBertEmbedding(config.embedding)

    embedding.fit(sample_text)
    print(embedding.transform(sample_text))
