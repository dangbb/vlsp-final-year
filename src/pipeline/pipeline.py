import os

import pandas as pd
from tqdm import tqdm

from src.config.config import Config
from src.evaluate.rouge_evaluator import ScoreSummary
from src.loader.class_loader import Dataset
from src.utils.factory import create_model, create_evaluator


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.df = pd.DataFrame(columns=[
            'cluster_id',
            'score',
        ])

    def training(self, dataset: Dataset):
        print("Start training in pipeline: ", self.config.name)

        # init
        dataset.set_source(self.config.source)
        if self.config.reset:
            self.df = pd.DataFrame(columns=[
                'cluster_id',
                'score',
            ])
        run_name = '{}_{}_{}'.format(
            self.config.name,
            self.config.model_config['name'],
            '_'.join(map(str, self.config.model_config['params'])),
        )

        local_result_path = os.path.join(self.config.result_path, run_name)
        if not os.path.exists(local_result_path):
            os.mkdir(local_result_path)

        score_summary = ScoreSummary(run_name)

        # create model
        model = create_model(self.config)

        # create evaluator
        evaluator = create_evaluator(self.config)

        # running
        try:
            print("Start training")
            for cluster in tqdm(dataset.clusters):
                summary, score = model.predict(cluster)

                self.df = self.df.append({
                    'cluster_id': cluster.cluster_idx,
                    'score': score,
                }, ignore_index=True)

                score_summary.add_score(
                    cluster.cluster_idx,
                    evaluator(
                        '.'.join(summary),
                        '.'.join(cluster.get_summary()),
                    )
                )
            print("Training complete")

            score_summary.save_report(local_result_path)
            self.df.to_csv(
                os.path.join(
                    local_result_path,
                    run_name + '-score.csv'
                ),
                index=False
            )
        except Exception as e:
            print("Training failed: ", e)
