import os
import logging

from sklearn.feature_extraction.text import CountVectorizer

from src.config.config import load_config_from_json
from src.loader.class_loader import Dataset
from src.loader.stopword_loader import stopword_reader

def get_stats_by_topic(dataset: Dataset, saved_path: str, word_frequency_folder_name: str):
    logging.warning("[JOB] - start calc stat by topic ...")
    import spacy
    nlp = spacy.load('vi_core_news_lg')

    stopwords = stopword_reader("/home/dang/vlsp-final-year/data/stopword/vietnamese.txt")

    topics = []
    for cluster in dataset.clusters:
        if cluster.category not in topics:
            topics.append(cluster.category)

    dataset.set_source('sent_splitted_token')

    for topic in topics:
        vectorizer = CountVectorizer(max_features=150)
        ultimate_cluster_term_count = []

        for cluster in dataset.clusters:
            if cluster.category == topic:
                ultimate_cluster_term_count = ultimate_cluster_term_count + cluster.get_all_sents()

        new_ultimate_cluster_term_count = []
        for sent in ultimate_cluster_term_count:
            tokens = [token for token in sent.split(' ') if token not in stopwords]
            new_ultimate_cluster_term_count.append(' '.join(tokens))

        ultimate_cluster_term_count = new_ultimate_cluster_term_count

        count = vectorizer.fit_transform([' '.join(ultimate_cluster_term_count)]).toarray()[0]

        count_term_combine = []
        for i, term in enumerate(vectorizer.get_feature_names_out()):
            count_term_combine.append((term, count[i]))

        count_term_combine = sorted(count_term_combine, key=lambda x: x[1], reverse=True)

        logging.warning("Topic: {}".format(topic))

        if not os.path.exists(os.path.join(saved_path, word_frequency_folder_name)):
            os.mkdir(os.path.join(saved_path, word_frequency_folder_name))

        saved_all_term_file = '{}_{}.{}'.format(topic, 'all', 'txt')
        with open(os.path.join(saved_path, word_frequency_folder_name, saved_all_term_file), "w") as f:
            for record in count_term_combine:
                f.write(record[0] + '\n')

        dataset.set_source('raw_text')


        only_verb = []
        only_noun = []
        only_abj = []

        for sent in ultimate_cluster_term_count:
            try:
                if sent.strip() == '':
                    continue
                doc = nlp(sent)
                for token in doc:
                    if token.tag_ == 'N':
                        only_noun.append(token.text)
                    elif token.tag_ == 'V':
                        only_verb.append(token.text)
                    elif token.tag_ == 'A':
                        only_abj.append(token.text)
            except Exception as e:
                logging.error("Error: sentence: {} - log: {}".format(sent, e))

        ## statistic for noun
        vectorizer = CountVectorizer(max_features=150)
        count = vectorizer.fit_transform([' '.join(only_noun)]).toarray()[0]
        count_term_combine = []
        for i, term in enumerate(vectorizer.get_feature_names_out()):
            count_term_combine.append((term, count[i]))
        count_term_combine = sorted(count_term_combine, key=lambda x: x[1], reverse=True)

        logging.warning("Topic: {} - Only noun".format(topic))

        saved_all_term_file = '{}_{}.{}'.format(topic, 'noun', 'txt')
        with open(os.path.join(saved_path, word_frequency_folder_name, saved_all_term_file), "w") as f:
            for record in count_term_combine:
                f.write(record[0] + '\n')

        ## statistic for verb
        vectorizer = CountVectorizer(max_features=150)
        count = vectorizer.fit_transform([' '.join(only_verb)]).toarray()[0]
        count_term_combine = []
        for i, term in enumerate(vectorizer.get_feature_names_out()):
            count_term_combine.append((term, count[i]))
        count_term_combine = sorted(count_term_combine, key=lambda x: x[1], reverse=True)

        logging.warning("Topic: {} - Only verb".format(topic))

        saved_all_term_file = '{}_{}.{}'.format(topic, 'verb', 'txt')
        with open(os.path.join(saved_path, word_frequency_folder_name, saved_all_term_file), "w") as f:
            for record in count_term_combine:
                f.write(record[0] + '\n')

        ## statistic for adj
        vectorizer = CountVectorizer(max_features=150)
        count = vectorizer.fit_transform([' '.join(only_abj)]).toarray()[0]
        count_term_combine = []
        for i, term in enumerate(vectorizer.get_feature_names_out()):
            count_term_combine.append((term, count[i]))
        count_term_combine = sorted(count_term_combine, key=lambda x: x[1], reverse=True)

        logging.warning("Topic: {} - Only adj".format(topic))

        saved_all_term_file = '{}_{}.{}'.format(topic, 'adj', 'txt')
        with open(os.path.join(saved_path, word_frequency_folder_name, saved_all_term_file), "w") as f:
            for record in count_term_combine:
                f.write(record[0] + '\n')

        logging.warning("[JOB] - done calc stat by topic ...")

if __name__ == '__main__':
    from src.loader.class_loader import load_cluster

    config = load_config_from_json()

    train_set = load_cluster(
        config.train_path
    )

    get_stats_by_topic(train_set, "/home/dang/vlsp-final-year/data", "term_freq")
