# plug/__init__.py
__version__ = "0.1.0"

from .embedding import embed_dataset
from .extraction import extract_token_vectors
from .model import PlugClassifier, cross_validate, fit, save_artifacts, predict