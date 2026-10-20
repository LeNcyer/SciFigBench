"""SciFigBench data, visual inputs and evaluation. Importing never downloads data."""
from ._version import __version__
from .data import DataError, PredictionError, load_qa, load_predictions
from .download import DownloadError, download_data
from .evaluation import evaluate
from .rendering import render, prepare_model_payload

__all__ = ["__version__", "download_data", "evaluate", "render", "prepare_model_payload", "load_qa", "load_predictions", "DataError", "PredictionError", "DownloadError"]
