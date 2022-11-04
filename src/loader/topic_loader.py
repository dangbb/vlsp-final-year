from typing import List

import os
import logging

from src.config.config import Config, load_config_from_json


def topic_loader(config: Config) -> List[str]:
    TOPIC_WORD_PATH = config.topic_path

    logging.warning("[Job {}] - Loading topic word...".format("{load topic}"))

    if not os.path.exists(TOPIC_WORD_PATH):
        raise Exception("topic path {} not exist".format(TOPIC_WORD_PATH))

    dirs = os.listdir(TOPIC_WORD_PATH)

    words_by_topic = {}
    count_topic = 0

    for dir in dirs:
        name = dir.split('.')[0]
        topic = name.split('_')[0]

        with open(os.path.join(TOPIC_WORD_PATH, dir)) as f:
            words = f.readlines()
            words = [word.strip('\n') for word in words]
            if topic not in words_by_topic.keys():
                words_by_topic[topic] = []
                count_topic += 1
            words_by_topic[topic] += words

    logging.warning("[Job {}] - Topic words loaded successfully. Total topic: {}".format("{load stopword}", count_topic))
    return words_by_topic


if __name__ == "__main__":
    config = load_config_from_json()
    words = topic_loader(config)
    print(words["Thế giới"])
