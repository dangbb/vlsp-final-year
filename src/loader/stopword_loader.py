from typing import List

import logging


def stopword_reader(path: str) -> List[str]:
    logging.warning("[Job {}] - Loading stopword...".format("{load stopword}"))
    with open(path, "r") as f:
        words = f.readlines()

    logging.warning("[Job {}] - Stopword loaded successfully. Total stopword: {}".format("{load stopword}", len(words)))
    return [word.strip('\n') for word in words]

if __name__ == "__main__":
    words = stopword_reader("/home/dang/vlsp-final-year/data/stopword /vietnamese.txt")
    print(words[:4])
