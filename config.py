"""
config.py
---------
Global constants, shared imports, and utility functions used across the pipeline.
Import this module at the top of every other script.
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Optional domain-specific libraries
# ---------------------------------------------------------------------------

ASE_AVAILABLE = False
PYM_AVAILABLE = False
SHAP_AVAILABLE = False

try:
    from ase.data import atomic_masses
    ASE_AVAILABLE = True
except Exception:
    pass

try:
    from pymatgen.core.structure import Structure
    PYM_AVAILABLE = True
except Exception:
    pass

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    pass

# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42       # Seed for all random operations — ensures reproducibility
DATA_PATH    = "dataset1.pckl.gz"  # Path to the pickled materials dataset
CV_FOLDS     = 5        # Number of folds for cross-validation
N_JOBS       = -1       # Use all available CPU cores

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def evaluate_regression(y_true, y_pred, name: str = "model") -> dict:
    """
    Compute MAE, RMSE, and R² for a set of predictions.

    Parameters
    ----------
    y_true : array-like  True target values.
    y_pred : array-like  Predicted target values.
    name   : str         Label printed alongside metrics.

    Returns
    -------
    dict with keys MAE, MSE, RMSE, R2.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    print(f"{name} → MAE={mae:.6f}  RMSE={rmse:.6f}  R²={r2:.6f}")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def crossval_report(pipe, X, y, cv: int = CV_FOLDS) -> dict:
    """
    Run stratified K-Fold cross-validation and return mean ± std of metrics.

    Parameters
    ----------
    pipe : sklearn Pipeline  The model pipeline to evaluate.
    X    : DataFrame         Feature matrix.
    y    : Series            Target vector.
    cv   : int               Number of folds.

    Returns
    -------
    dict with MAE_mean, MAE_std, RMSE_mean, RMSE_std, R2_mean, R2_std.
    """
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "MSE": "neg_mean_squared_error",
        "R2":  "r2",
    }
    res  = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=N_JOBS)
    mae  = -res["test_MAE"]
    rmse = np.sqrt(-res["test_MSE"])
    r2   = res["test_R2"]
    return {
        "MAE_mean":  mae.mean(),  "MAE_std":  mae.std(),
        "RMSE_mean": rmse.mean(), "RMSE_std": rmse.std(),
        "R2_mean":   r2.mean(),   "R2_std":   r2.std(),
    }
