from src.config.config import Config, ModelConfig, EvalConfig
from src.evaluate.rouge_evaluator import PipRouge
from src.model.lexrankCustom import Lexrank
from src.model.mmr import MMR
from src.model.pyramidExtraction import PyramidExt
from src.model.textrank import TextrankCustom
from src.model.viT5 import ViT5


def create_model(config: ModelConfig):
    """
    Return a model, based on config.
    Default: Textrank Custom
    :param config:
    :return:
    """
    if config.name == 'textrank':
        model = TextrankCustom(config)
        return model
    elif config.name == 'mmr':
        model = MMR(config)
        return model
    elif config.name == 'lexrank':
        model = Lexrank(config)
        return model
    elif config.name == 'vit5':
        model = ViT5(config)
        return model
    elif config.name == 'pyramid':
        model = PyramidExt(config)
        return model
    else:
        raise Exception('Unsupported model {}'.format(config.name))


def create_evaluator(config: EvalConfig):
    """
    Return an evaluator, based on config.
    Default: Pip rouge, support rouge-1, rouge-2, rouge-l.
    :param config:
    :return:
    """
    if config.name == 'pip_rouge':
        evaluator = PipRouge()
        return evaluator
    else:
        raise Exception('Unsupported evaluator {}'.format(config.name))
