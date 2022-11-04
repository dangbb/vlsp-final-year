from src.config.config import Config, ModelConfig, EvalConfig
from src.evaluate.rouge_evaluator import PipRouge
from src.model.combination_model.textrank_mmr import TextRank_MMR_Model
from src.model.customMLP import CustomMLP
from src.model.lexrankBertEmbedding import LexRankBertModel
from src.model.lexrankCustom import Lexrank
from src.model.mmr import MMR
from src.model.mmr_query import MMRQuery
from src.model.mmr_query_anchor import MMRQueryAnchor
from src.model.mmr_query_best_title import MMRQueryBestTitle
from src.model.random_model import RandomModel
from src.model.textrank import TextrankCustom


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
    elif config.name == 'mlp':
        model = CustomMLP(config)
        return model
    elif config.name == 'random':
        model = RandomModel(config)
        return model
    elif config.name == 'textrank_mmr':
        model = TextRank_MMR_Model(config)
        return model
    elif config.name == 'textrank_bert':
        model = LexRankBertModel(config)
        return model
    elif config.name == 'mmr_query':
        model = MMRQuery(config)
        return model
    elif config.name == 'mmr_query_best_title':
        model = MMRQueryBestTitle(config)
        return model
    elif config.name == 'mmr_query_anchor':
        model = MMRQueryAnchor(config)
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
