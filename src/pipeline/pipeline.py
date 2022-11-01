import logging
import os

import pandas as pd
from tqdm import tqdm

from src.config.config import Config, load_config_from_json
from src.evaluate.rouge_evaluator import ScoreSummary
from src.loader.class_loader import Dataset
from src.utils.factory import create_model, create_evaluator


class Pipeline:
    def __init__(self, config: Config, model_idx: int = 0):
        self.local_result_path = None
        self.config = config
        self.train_df = pd.DataFrame(columns=[
            'cluster_id',
            'score',
        ])
        self.valid_df = pd.DataFrame(columns=[
            'cluster_id',
            'score',
        ])
        self.test_df = pd.DataFrame(columns=[
            'cluster_id',
            'score',
        ])
        self.model_config = config.models[model_idx]
        self.eval_config = config.eval_config

    def save_result_txt(self, res_name):
        """
        Save the result of the pipeline in a txt file
        :return: none
        """
        result_series = self.test_df["summary"]
        with open(os.path.join(self.local_result_path, "results.txt"), "w") as f:
            for summary in result_series:
                f.write(summary)
                f.write('\n')

    def get_run_name(self):
        run_name = '{}_{}_{}_{}'.format(
            self.config.name,
            self.model_config.name,
            '_'.join(map(str, self.model_config.params)),
            self.model_config.document_convention,
        )

        return run_name

    def create_result_directory(self):
        self.local_result_path = os.path.join(self.config.result_path, self.get_run_name())
        if not os.path.exists(self.local_result_path):
            os.mkdir(self.local_result_path)

    def training(
            self,
            dataset: Dataset,
            valid_dataset: Dataset = None,
    ):
        print("Start training in pipeline: ", self.config.name)

        # init
        dataset.set_source(self.model_config.source)
        if self.config.reset:
            self.train_df = pd.DataFrame(columns=[
                'cluster_id',
                'score',
                'summary',
            ])
            self.valid_df = pd.DataFrame(columns=[
                'cluster_id',
                'score',
                'summary',
            ])
            self.test_df = pd.DataFrame(columns=[
                'cluster_id',
                'score',
                'summary',
            ])

        run_name = self.get_run_name()
        training_run_name = run_name + '_train'
        validate_run_name = run_name + '_valid'

        self.create_result_directory()

        training_score_summary = ScoreSummary(training_run_name)
        validate_score_summary = ScoreSummary(validate_run_name)

        # create model
        model = create_model(self.model_config)

        # create evaluator
        evaluator = create_evaluator(self.eval_config)

        if self.model_config.training_required:
            # is training_required, model is training on full dataset before predicting
            model.training(dataset)

        # running
        try:
            logging.warning("[JOB] '{}' - Start training.".format(training_run_name))
            for cluster in tqdm(dataset.clusters):
                summary, score = model.predict(cluster)

                self.train_df = self.train_df.append({
                    'cluster_id': cluster.cluster_idx,
                    'score': score,
                    'summary': '.'.join(summary),
                }, ignore_index=True)

                training_score_summary.add_score(
                    cluster.cluster_idx,
                    evaluator(
                        '.'.join(summary),
                        '.'.join(cluster.get_summary()),
                    )
                )
            logging.warning("[JOB] '{}' - Training complete. Saving report.".format(training_run_name))

            training_score_summary.save_report(self.local_result_path)
            self.train_df.to_csv(
                os.path.join(
                    self.local_result_path,
                    training_run_name + '-score.csv'
                ),
                index=False
            )

            logging.warning("[JOB] '{}' - Saving report complete.".format(run_name))

            # evaluate in valid set
            if valid_dataset is not None:
                logging.warning("[JOB] '{}' - Start training on validate set.".format(validate_run_name))

                for cluster in tqdm(valid_dataset.clusters):
                    summary, score = model.predict(cluster)

                    self.valid_df = self.valid_df.append({
                        'cluster_id': cluster.cluster_idx,
                        'score': score,
                        'summary': '.'.join(summary),
                    }, ignore_index=True)

                    validate_score_summary.add_score(
                        cluster.cluster_idx,
                        evaluator(
                            '.'.join(summary),
                            '.'.join(cluster.get_summary()),
                        )
                    )

                logging.warning("[JOB] '{}' - Training on valid set complete. Saving report.".format(validate_run_name))

                validate_score_summary.save_report(self.local_result_path)
                self.valid_df.to_csv(
                    os.path.join(
                        self.local_result_path,
                        validate_run_name + '-score.csv'
                    ),
                    index=False
                )

                logging.warning("[JOB] '{}' - Saving valid report complete.".format(validate_run_name))
            else:
                logging.warning("[JOB] '{}' - Valid dataset not found.".format(validate_run_name))

            # evaluate in test set

        except Exception as e:
            print("Training failed: ", e)

    def predict(self, test_dataset):
        testing_run_name = self.get_run_name() + '_test'
        self.create_result_directory()

        model = create_model(self.model_config)

        if test_dataset is not None:
            logging.warning("[JOB] '{}' - Start predict on test.".format(testing_run_name))

            for cluster in tqdm(test_dataset.clusters):
                summary, score = model.predict(cluster)

                self.test_df = self.test_df.append({
                    'cluster_id': cluster.cluster_idx,
                    'score': score,
                    'summary': '.'.join(summary),
                }, ignore_index=True)

            logging.warning("[JOB] '{}' - Predict on test complete. Saving report.".format(testing_run_name))

            self.test_df.to_csv(
                os.path.join(
                    self.local_result_path,
                    testing_run_name + '-score.csv'
                ),
                index=False
            )

            self.save_result_txt(testing_run_name)

            logging.warning("[JOB] '{}' - Saving test report complete.".format(testing_run_name))
        else:
            logging.warning("[JOB] '{}' - Test dataset not found.".format(testing_run_name))


if __name__ == '__main__':
    from src.loader.class_loader import load_cluster

    config = load_config_from_json()

    # train_set = load_cluster(
    #     config.train_path
    # )
    # valid_set = load_cluster(
    #     config.valid_path
    # )
    test_set = load_cluster(
        config.test_path
    )

    pipeline0 = Pipeline(config, 0)
    pipeline0.predict(test_set)

    # pipeline0 = Pipeline(config, 1)
    # pipeline0.training(train_set, valid_set)
    #
    # pipeline0 = Pipeline(config, 2)
    # pipeline0.training(train_set, valid_set)
    #
    # pipeline0 = Pipeline(config, 3)
    # pipeline0.training(train_set, valid_set)
