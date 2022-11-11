from rouge import Rouge

import numpy as np
from tqdm import tqdm

from src.config.config import Config
from src.loader.class_loader import Cluster

N_PERCENT = .1


def get_score(predict, ref):
    predict = predict.replace('_', ' ')
    ref = ref.replace('_', ' ')
    rouge = Rouge()

    result = rouge.get_scores(predict, ref)

    return result[0]['rouge-1']['p'], result[0]['rouge-1']['r'], result[0]['rouge-1']['f']


def get_label(cluster: Cluster):
    cluster_label = []
    scores = []

    for idx, sent in tqdm(enumerate(cluster.get_all_sents()), "Extract score"):
        best_score = 0.0
        for golden in cluster.get_summary():
            if get_score(sent, golden)[0] >= best_score:
                best_score = get_score(sent, golden)[0]
        scores.append(best_score)

    scores = sorted(scores, reverse=True)
    threshold = max(.5, scores[int(len(scores) * N_PERCENT)])

    for idx, sent in tqdm(enumerate(cluster.get_all_sents()), "Extract label"):
        if scores[idx] >= threshold:
            cluster_label.append(1)
        else:
            cluster_label.append(0)

    return cluster_label


if __name__ == "__main__":
    from src.loader.class_loader import load_cluster
    train_set = load_cluster(
        Config.load_config_from_json().train_path, 1
    )
    valid_set = load_cluster(
        Config.load_config_from_json().valid_path, 1
    )

    for cluster in train_set.clusters:
        label = get_label(cluster)
        print(len(label))
