from typing import List


class EvalConfig:
    def __init__(self):
        self.name = ''


class EmbeddingConfig:
    def __init__(self):
        self.model = ''
        self.distance = ''
        self.max_df = ''
        self.bart_path = ''


class ModelConfig:
    def __init__(self):
        self.name = ''
        self.count_word = True,
        self.params = []
        self.n_words = 0
        self.sigma = 0
        self.embedding: EmbeddingConfig = EmbeddingConfig()
        self.training_required = False
        self.document_convention = ""
        self.source = ""


class Config:
    def __init__(self):
        self.name: str = ''
        self.train_path: str = ''
        self.valid_path: str = ''
        self.test_path: str = ''
        self.result_path = ''

        self.eval_config = {}

        self.reset = True
        self.DEBUG = False

        self.models: List[ModelConfig] = []
        self.embedding: EmbeddingConfig = EmbeddingConfig()


def parse_eval_config(data) -> EvalConfig:
    eval_config = EvalConfig()
    eval_config.name = data['name']
    return eval_config


def parse_embedding_config(data) -> EmbeddingConfig:
    embedding_config = EmbeddingConfig()
    embedding_config.model = data['model']
    embedding_config.distance = data['distance']
    embedding_config.max_df = data['max_df']
    embedding_config.bart_path = data['bart_path']
    return embedding_config


def parse_model_config(data) -> ModelConfig:
    model_config = ModelConfig()
    model_config.source = data['source']
    model_config.name = data['name']
    model_config.count_word = data['count_word']
    model_config.params = data['params']
    model_config.n_words = data['n_words']
    model_config.sigma = data['sigma']
    model_config.embedding = parse_embedding_config(
        data['embedding']
    )
    model_config.training_required = data['training_required']
    model_config.document_convention = data['document_convention']
    return model_config


def load_config_from_json(
        path: str = '/home/hvn/Documents/dskt/vlsp-final-year/env.json'
) -> Config:
    import json

    with open(path) as f:
        data = json.load(f)
        config = Config()
        config.name = data['name']
        config.train_path = data['train_path']
        config.valid_path = data['valid_path']
        config.test_path = data['test_path']
        config.models = [
            parse_model_config(model_data)
            for model_data in data['models']
        ]
        config.eval_config = parse_eval_config(
            data['eval']
        )
        config.embedding = parse_embedding_config(
            data['embedding']
        )
        config.reset = data['reset']
        config.result_path = data['result_path']
        config.debug = data['debug']

        return config


if __name__ == "__main__":
    config = load_config_from_json()
