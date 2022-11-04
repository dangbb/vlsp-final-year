import logging
import argparse

from src.config.config import  load_config_from_json
from src.loader.class_loader import load_cluster
from src.pipeline.pipeline import Pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-path", help="config path for pipeline", required=True)
    parser.add_argument("-f", "--train-path", help="train path")
    parser.add_argument("-v", "--valid-path", help="valid path")
    parser.add_argument("-e", "--eval-path", help="eval path")
    parser.add_argument("-m", "--model-index", help="index of model in config file", nargs="+", type=int)

    args = parser.parse_args()

    config_path = args.config_path
    train_path = args.train_path
    valid_path = args.valid_path
    test_path = args.eval_path

    print("index set: ", args.model_index)

    index_set = list(set(args.model_index))

    print("Arg parse: ")
    print("Config path: ", config_path)
    print("Train path: ", train_path)
    print("Valid path: ", valid_path)
    print("Test path: ", test_path)
    print("Index set: ", index_set)

    config = load_config_from_json(config_path)

    # train_set = load_cluster(
    #     config.train_path
    # )
    try:
        train_set = load_cluster(
            train_path
        )
        logging.warning("[PIPELINE] - Load train set from {}. Done.".format(train_path))
    except Exception as e:
        train_set = None
        logging.warning("[PIPELINE] - Load train set from {}. Failed. Using None.".format(train_path))

    try:
        valid_set = load_cluster(
            valid_path
        )
        logging.warning("[PIPELINE] - Load valid set from {}. Done.".format(train_path))
    except Exception as e:
        valid_set = None
        logging.warning("[PIPELINE] - Load valid set from {}. Failed. Using None.".format(train_path))

    try:
        test_set = load_cluster(
            test_path
        )
        logging.warning("[PIPELINE] - Load test set from {}. Done.".format(train_path))
    except Exception as e:
        test_set = None
        logging.warning("[PIPELINE] - Load test set from {}. Failed. Using None.".format(train_path))

    for idx in index_set:
        try:
            pipeline0 = Pipeline(config, idx)
            try:
                pipeline0.training(train_set, valid_set)
            except Exception as e:
                print(e)
            pipeline0.predict(test_set)
        except Exception as e:
            logging.error("[PIPELINE] - Init pipeline failed, error: ", e)