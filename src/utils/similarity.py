from numpy import dot, array
from numpy.linalg import norm

from src.config.config import EmbeddingConfig


def cosine_similarity(vector_a: array, vector_b: array):
    return dot(vector_a, vector_b) / (norm(vector_a) * norm(vector_b))


def distance_similarity(vector_a: array, vector_b: array):
    return norm(vector_a - vector_b)


def get_similarity(config: EmbeddingConfig):
    if config.distance == 'cosine':
        return cosine_similarity
    else:
        return distance_similarity
