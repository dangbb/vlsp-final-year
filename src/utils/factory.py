from src.config.config import Config
from src.evaluate.rouge_evaluator import PipRouge
from src.model.textrank import TextrankCustom


def create_model(config: Config):
    """
    Return a model, based on config.
    Default: Textrank Custom
    :param config:
    :return:
    """
    if config.model_config['name'] == 'textrank':
        model = TextrankCustom(config)
        return model
    else:
        raise Exception('Unsupported model {}'.format(config.model_config['name']))


def create_evaluator(config: Config):
    """
    Return an evaluator, based on config.
    Default: Pip rouge, support rouge-1, rouge-2, rouge-l.
    :param config:
    :return:
    """
    if config.eval_config['name'] == 'pip_rouge':
        evaluator = PipRouge()
        return evaluator
    else:
        raise Exception('Unsupported evaluator {}'.format(config.eval_config['name']))
