import logging
import numpy as np

from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from underthesea import ner

from external.sentence_transformer.sentence_transformers.sentence_transformers import SentenceTransformer
from src.config.config import EmbeddingConfig, load_config_from_json, Config
from src.evaluate.rouge_evaluator import PipRouge
from src.loader.class_loader import Dataset, Cluster, load_cluster, SOURCE
from src.loader.stopword_loader import stopword_reader
from src.loader.topic_loader import topic_loader


class Embedding:
    def __init__(self):
        pass

    def fit(self, sentences: List[str]):
        pass

    def fit_dataset(self, dataset: Dataset):
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


def count_ner(sent: str):
    ners = ner(sent)

    total = 0

    for token in ners:
        tag = token[3]
        if tag[0] == 'B':
            total += 1

    return total


def count_postag(sent: str):
    from underthesea import pos_tag

    postags = pos_tag(sent)
    postag_total = 0

    word_count = {
        'N': 0,
        'V': 0,
        'A': 0,
    }

    for token in postags:
        tag = token[1]
        if tag in ['N', 'V', 'A']:
            word_count[tag] += 1
            postag_total += 1

    return word_count, postag_total

class SentenceCustomEmbedding(Embedding):
    def __init__(self, config: Config):
        super(SentenceCustomEmbedding, self).__init__()
        logging.warning(' - Init sentence embedding instance')

        self.config = config

        # new feature must be registed here
        self.columns = [
            'tf-idf',
            'len',
            'ner-total',
            'postag-total',
            'postag-N',
            'postag-V',
            'postag-A',
            'topic_word_count',
            'label'
        ]

        # new feature dependency must be registed here
        ## topic word
        self.stopword_list = stopword_reader(config)

        ## stopword
        self.topic_word = topic_loader(config)

        ## tf-idf
        self.tf_idf = TfidfVectorizer(
            max_df=0.5,
        )
        self.tf_idf_trained = False

        ## evaluator
        self.evaluator = PipRouge()

    def fit(self, sentences: List[str]):
        # some feature just need to be trained on
        pass

    def fit_dataset(self, dataset: Dataset):
        # some feature need to be trained on dataset
        ## tf-idf
        dataset.set_source(SOURCE.SENT_SPLITTED_TOKEN.value)
        corpus = []

        for cluster in dataset.clusters:
            corpus.append(' '.join(cluster.get_all_sents()))

        self.tf_idf.fit(corpus)

        self.tf_idf_trained = True

    def count_by_topic(self, sent: List[str], topic_name: str):
        if topic_name not in self.topic_word.keys():
            logging.error('topic name {} not found'.format(topic_name))
            return 0

        count = 0
        for word in sent.split(' '):
            if word in self.topic_word[topic_name]:
                count += 1
        return count

    def calc_rouge(self, sent: str, cluster: Cluster):
        summary = '.'.join(cluster.get_summary())

        rouge_score = self.evaluator(sent, summary)

        return rouge_score.rougeL.p

    def transform_cluster(self, cluster: Cluster):
        embedding_batch = pd.DataFrame(columns=self.columns)
        clone_cluster = cluster
        clone_cluster.set_source(SOURCE.SENT_SPLITTED_TOKEN.value)

        for sent in cluster.get_all_sents():
            embedding = {}

            for col in self.columns:
                embedding[col] = 0.0

            try:
                ## tf-idf handler
                if self.tf_idf_trained == False:
                    logging.warning('tf-idf hasnt been trained yet -> disable tf-idf')

                    embedding['tf-idf'] = 0.0
                else:
                    tfidf_vector = self.tf_idf.transform([sent]).toarray()[0]
                    tfidf_vector = sorted(tfidf_vector, reverse=True)
                    tfidf_score = sum(tfidf_vector[:min(10, len(tfidf_vector))]) / min(10, len(tfidf_vector))

                    embedding['tf-idf'] = float(tfidf_score)

                ## length
                embedding['len'] = np.log(len(sent.split(' ')))

                ## ner-total
                count_ner_total = count_ner(sent)
                embedding['ner-total'] = count_ner_total

                ## postag
                count_postag_each, count_postag_total = count_postag(sent)
                embedding['postag-total'] = count_postag_total
                for key in count_postag_each.keys():
                    embedding['postag-{}'.format(key)] = count_postag_each[key]

                ## topic word count
                embedding['topic_word_count'] = self.count_by_topic(sent, cluster.category)

                # get label
                embedding['label'] = 0

                for key in embedding.keys():
                    try:
                        value = float(embedding[key])
                    except Exception as e:
                        print(e)
                        print(key)
                        print(embedding[key])

                # push to dataframe
                embedding_batch = embedding_batch.append(embedding, ignore_index=True)

            except Exception as e:
                logging.error('error when transforming, err: ', e)
                break
        assert len(embedding_batch) == len(cluster.get_all_sents()), "mismatch size, embedding {} != n len {}".format(len(embedding_batch), len(cluster.get_all_sents()))

        return embedding_batch

    def transform_dataset(self, dataset: Dataset):
        dataset.set_source(SOURCE.SENT_SPLITTED_TOKEN.value)
        embedding_batch = pd.DataFrame(columns=self.columns)

        logging.warning('[JOB {}] - start transform sentences to embedding vector'.format('vector embedding'))

        for idx, cluster in tqdm(enumerate(dataset.clusters)):
            for sent in cluster.get_all_sents():
                embedding = {}

                for col in self.columns:
                    embedding[col] = 0.0

                try:
                    ## tf-idf handler
                    if self.tf_idf_trained == False:
                        logging.warning('tf-idf hasnt been trained yet -> disable tf-idf')

                        embedding['tf-idf'] = 0.0
                    else:
                        tfidf_vector = self.tf_idf.transform([sent]).toarray()[0]
                        tfidf_vector = sorted(tfidf_vector, reverse=True)
                        tfidf_score = sum(tfidf_vector[:min(10, len(tfidf_vector))]) / min(10, len(tfidf_vector))

                        embedding['tf-idf'] = float(tfidf_score)

                    ## length
                    embedding['len'] = np.log(len(sent.split(' ')))

                    ## ner-total
                    count_ner_total = count_ner(sent)
                    embedding['ner-total'] = count_ner_total

                    ## postag
                    count_postag_each, count_postag_total = count_postag(sent)
                    embedding['postag-total'] = count_postag_total
                    for key in count_postag_each.keys():
                        embedding['postag-{}'.format(key)] = count_postag_each[key]

                    ## topic word count
                    embedding['topic_word_count'] = self.count_by_topic(sent, cluster.category)

                    # get label
                    embedding['label'] = self.calc_rouge(sent, cluster)

                    for key in embedding.keys():
                        try:
                            value = float(embedding[key])
                        except Exception as e:
                            print(e)
                            print(key)
                            print(embedding[key])

                    # push to dataframe
                    embedding_batch = embedding_batch.append(embedding, ignore_index=True)

                except Exception as e:
                    logging.error('error when transforming, err: ', e)
                    break

        logging.warning('[JOB {}] - transform sentences to embedding vector. done.'.format('vector embedding'))

        return embedding_batch

def get_embedding(config: EmbeddingConfig) -> Embedding:
    if config.model == 'SBERT':
        return SentenceBertEmbedding(config)
    elif config.model == 'tf-idf':
        return TfIdfEmbedding(config)
    else:
        raise Exception('Unsupported embedding {}'.format(config.model))


if __name__ == "__main__":
    config = load_config_from_json()

    dataset = load_cluster(config.train_path, 1)
    dataset.set_source(SOURCE.SENT_SPLITTED_TOKEN.value)
    total = 0
    for cluster in dataset.clusters:
        total += len(cluster.get_all_sents())

    print("input shape: ", total)

    embedding = SentenceCustomEmbedding(config)
    embedding.fit_dataset(dataset)

    df = embedding.transform_cluster(dataset.clusters[0])
    print("shape: ", df.shape)
    print("col: ", df.columns)

    # test on sklearn model
    from sklearn.neural_network import MLPRegressor

    model = MLPRegressor()
    model.fit(df, np.array(df['label']).astype(float))
    model.predict(df)

