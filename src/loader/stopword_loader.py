import os.path
from typing import List

import logging

from src.config.config import Config


def stopword_reader(config: Config) -> List[str]:
    stopword_path = config.stopword_path
    logging.warning("[Job {}] - Loading stopword from {}".format("{load stopword}", stopword_path))
    if not os.path.exists(stopword_path):
        raise Exception('Folder {} for found '.format(stopword_path))

    words = []

    for filename in os.listdir(config.stopword_path):
        dir = os.path.join(config.stopword_path, filename)

        logging.warning("[Job {}] - Loading stopword...".format("{load stopword}"))
        with open(dir, "r") as f:
            words = words + f.readlines()

    logging.warning("[Job {}] - Stopword loaded successfully. Total stopword: {}".format("{load stopword}", len(words)))
    return [word.strip('\n') for word in words]

if __name__ == "__main__":
    config = Config.load_config_from_json()
    words = stopword_reader(config)
    print(words[:4])
