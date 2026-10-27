"""Classify terrain pixels with the optional LightGBM model."""
import numpy as np

from app import core


NAME = "LightGBM"


def fit_rows(Xtr, ytr, w=None):
    """Train on labeled pixels, using confidence weights when provided."""
    if Xtr is None or len(np.unique(ytr)) < 2:
        return None
    import lightgbm as lgb
    model = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.1, num_leaves=31,
                              min_child_samples=20, class_weight="balanced",
                              verbose=-1, random_state=0)
    model.fit(Xtr, ytr, sample_weight=w)
    return model


def predict(model, X):
    """Predict uint8 unit codes on the raster grid, using 255 for missing terrain."""
    return core.blockwise(model.predict, X)


def confidence(model, X):
    """Return float32 predicted-unit probabilities, using NaN for missing terrain."""
    return core.blockwise(lambda rows: model.predict_proba(rows).max(1), X,
                          dtype=np.float32, fill=np.nan)
