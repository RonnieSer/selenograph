"""Placeholder for a U-Net terrain-segmentation backend.

Implementation requires spatial training tiles, a loss that ignores unpainted
cells, and assistant storage that preserves spatial context instead of pooled
pixel rows. PyTorch and a segmentation implementation are also required.
"""
from app import core


NAME = "U-Net (not implemented)"

_MSG = ("The U-Net backend is a placeholder and has nothing behind it yet. "
        "Set MODEL = \"gbm\" (or \"rf\") in core.py, or implement model_unet.py -- "
        "the file documents what it needs.")


def fit(X, cells):
    """Raise NotImplementedError until spatial training is available."""
    raise NotImplementedError(_MSG)


def fit_rows(Xtr, ytr, w=None):
    """Raise NotImplementedError because pooled rows lack spatial context."""
    raise NotImplementedError(_MSG)


def predict(model, X):
    """Raise NotImplementedError until segmentation inference is available."""
    raise NotImplementedError(_MSG)


def confidence(model, X):
    """Raise NotImplementedError until segmentation probabilities are available."""
    raise NotImplementedError(_MSG)
