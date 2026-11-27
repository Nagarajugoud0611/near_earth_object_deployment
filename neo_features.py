"""
Shared feature engineering for the NEO (Near-Earth Object) project.

Kept in its own module so that:
  * the training scripts and the Streamlit app use the exact same code
  * the saved pipeline (pickle / cloudpickle) can find the function again when loaded

The pipeline takes the 5 RAW numeric columns and does the log transform +
feature engineering internally, so nobody has to repeat it at prediction time.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

RAW_FEATURES = [
    "est_diameter_min",
    "est_diameter_max",
    "relative_velocity",
    "miss_distance",
    "absolute_magnitude",
]


def engineer_features(X):
    """Raw columns -> log-transformed diameters + engineered features (Sprint 1 + Sprint 3)."""
    X = pd.DataFrame(X, columns=RAW_FEATURES).astype(float)

    d_min = X["est_diameter_min"]
    d_max = X["est_diameter_max"]
    d_avg = (d_min + d_max) / 2
    velocity = X["relative_velocity"]
    miss = X["miss_distance"]

    out = pd.DataFrame(index=X.index)
    # Sprint 1: log transform of the heavily skewed diameter columns
    out["est_diameter_min"] = np.log1p(d_min)
    out["est_diameter_max"] = np.log1p(d_max)
    out["relative_velocity"] = velocity
    out["miss_distance"] = miss
    out["absolute_magnitude"] = X["absolute_magnitude"]

    # Sprint 3: engineered features
    out["diameter_avg"] = np.log1p(d_avg)
    out["diameter_range"] = d_max - d_min
    out["kinetic_proxy"] = np.log1p(d_avg ** 3 * velocity ** 2)   # size x speed ~ impact energy
    out["miss_per_velocity"] = miss / velocity                     # rough time scale of the flyby
    out["diameter_to_miss"] = np.log1p(d_avg / miss)               # size relative to distance
    return out


def feature_engineer():
    """FunctionTransformer wrapper, used as the first step of every pipeline."""
    return FunctionTransformer(engineer_features, validate=False)
