"""LightGBM backend for per-pixel terrain classification.

Spatial context is represented by terrain features computed over local windows.
"""
import numpy as np

from app import core


NAME = "LightGBM"


def fit(X, cells):
    """Fit terrain units from a feature stack and painting cells.

    X has shape (n_features, height, width); cells has shape (GRID, GRID), with
    zero denoting unpainted cells. Return None if fewer than two units occur.
    """
    Xtr, ytr = core.cell_pixels(X, cells)
    return fit_rows(Xtr, ytr)


def fit_rows(Xtr, ytr, w=None):
    """Fit sampled feature rows with optional weights, requiring at least two units."""
    if Xtr is None or len(np.unique(ytr)) < 2:
        return None
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.1, num_leaves=31,
                             min_child_samples=20, class_weight="balanced", verbose=-1,
                             random_state=0)
    clf.fit(Xtr, ytr, sample_weight=w)
    return clf


def predict(model, X):
    """Predict uint8 unit codes on the raster grid, using 255 for missing terrain."""
    return core.blockwise(model.predict, X)


def confidence(model, X):
    """Return float32 predicted-unit probabilities, using NaN for missing terrain."""
    return core.blockwise(lambda rows: model.predict_proba(rows).max(1), X,
                          dtype=np.float32, fill=np.nan)
