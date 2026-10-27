"""Classify terrain pixels with a Random Forest."""
import numpy as np

from app import core


NAME = "Random forest"


def fit_rows(Xtr, ytr, w=None):
    """Train on labeled pixels, using confidence weights when provided."""
    if Xtr is None or len(np.unique(ytr)) < 2:
        return None
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                   class_weight="balanced", n_jobs=-1, random_state=0)
    model.fit(Xtr, ytr, sample_weight=w)
    return model


def predict(model, X):
    """Predict uint8 unit codes on the raster grid, using 255 for missing terrain."""
    return core.blockwise(model.predict, X)


def confidence(model, X):
    """Return float32 predicted-unit probabilities, using NaN for missing terrain."""
    return core.blockwise(lambda rows: model.predict_proba(rows).max(1), X,
                          dtype=np.float32, fill=np.nan)
