from tqdm.notebook import tqdm

import itertools

from src.loader.class_loader import Cluster, load_cluster
from src.config.config import load_config_from_json
from src.evaluate.rouge_evaluator import PipRouge
from src.evaluate.rouge_evaluator import ScoreSummary


class OracleConfig():
    def __init__(self):
        self.rouge = 'l'
        self.metric = 'p'
        self.candidate_sent = 10
        self.chosen_sen = 8


class OracleGenerator():
    def __init__(self, config: OracleConfig):
        self.config = config

    def generate_oracle(self, cluster: Cluster):
        summary = cluster.get_summary()
        all_sents = cluster.get_all_sents()
        rouge = PipRouge()

        scores = []

        for i, sent in enumerate(all_sents):
            try:
                rouge_score = rouge(sent, '.'.join(summary))

                if self.config.rouge == '1':
                    rouge_score = rouge_score.rouge1
                elif self.config.rouge == '2':
                    rouge_score = rouge_score.rouge2
                elif self.config.rouge == 'l':
                    rouge_score = rouge_score.rougeL
                else:
                    raise Exception("Invalid rouge type, expected ['1', '2', 'l'], get {}".format(self.rouge))

                if self.config.metric == 'p':
                    rouge_score = rouge_score.p
                elif self.config.metric == 'r':
                    rouge_score = rouge_score.r
                elif self.config.metric == 'f1':
                    rouge_score = rouge_score.f1
                else:
                    raise Exception("Invalid rouge metric, expected ['p', 'r', 'f1'], get {}".format(self.metric))

                scores.append((i, rouge_score))
            except Exception as e:
                print("Error: ", e)
                print("When: {} - {}", i, sent)
                scores.append((i, -99999))

        top_index = sorted(scores, key=lambda x: x[1], reverse=True)
        top_index = top_index[:self.config.candidate_sent]
        top_index = [idx[0] for idx in top_index]

        candidate_combinations = list(itertools.combinations(list(top_index), self.config.chosen_sen))
        candidate_score = []

        for combination in candidate_combinations:
            conbination_sent = [all_sents[idx] for idx in combination]
            conbination_docs = '.'.join(conbination_sent)

            try:
                rouge_score = rouge(conbination_docs, '.'.join(summary))

                candidate_score.append((combination, rouge_score))
            except Exception as e:
                print("Error: ", e)
                print("Hypothesis: ", conbination_docs)
                print("Label: ", '.'.join(summary))
                print("Current combination: ", combination)

        candidate_score = sorted(candidate_score, key=lambda x: x[1].rouge2.f1, reverse=True)
        oracle_combination = candidate_score[0][0]
        oracle_score = candidate_score[0][1]

        return oracle_combination, oracle_score

if __name__ == "__main__":
    config = load_config_from_json()

    dataset = load_cluster(
        config.train_path,
        1
    )
    dataset.set_source('sent_splitted_token')

    rouge_summary = ScoreSummary("golden_summary_eval-13-10-")

    oracle_config = OracleConfig()
    oracle_config.chosen_sen = 10
    oracle_config.candidate_sent = 13
    oracle_generator = OracleGenerator(oracle_config)

    for idx, cluster in tqdm(enumerate(dataset.clusters)):
        com, _ = oracle_generator.generate_oracle(dataset.clusters[0])
        print(com)

    # rouge_summary.save_report("/home/dang/vlsp-final-year/data/result")