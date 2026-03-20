"""
train_baseline.py
-----------------
Train four baseline models on the raw numeric features (no domain descriptors,
no hyperparameter tuning) to establish a performance floor.

Models
------
  LinearRegression  — simple linear model; fast, interpretable, limited capacity.
  RandomForest      — ensemble of decision trees; handles non-linearity well.
  GradientBoosting  — sequential boosting; strong out-of-the-box performance.
  SVR               — support vector regression; needs scaled features.

Each trained pipeline is saved as  baseline_<ModelName>.joblib.
A summary CSV is written to        baseline_metrics.csv.

Usage
-----
    python src/train_baseline.py

Prerequisite
------------
    The dataset file specified by DATA_PATH in config.py must be present.
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

from config import DATA_PATH, RANDOM_STATE, evaluate_regression


def train_baselines() -> None:
    # --- Load data (numeric columns only, no domain descriptors) ---
    df = pd.read_pickle(DATA_PATH)
    if "atoms" in df.columns:
        df = df.drop(columns=["atoms"])
    df = df.dropna(subset=["energy_per_atom"]).reset_index(drop=True)

    X = df.drop(columns=["energy_per_atom"]).select_dtypes(include=[np.number])
    y = df["energy_per_atom"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # --- Model definitions ---
    # SVR requires StandardScaler; the other models are scale-invariant.
    models = {
        "LinearRegression": Pipeline(
            [("model", LinearRegression())]
        ),
        "RandomForest": Pipeline(
            [("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1))]
        ),
        "GradientBoosting": Pipeline(
            [("model", GradientBoostingRegressor(random_state=RANDOM_STATE))]
        ),
        "SVR": Pipeline(
            [("scaler", StandardScaler()), ("model", SVR())]
        ),
    }

    # --- Train, evaluate, save ---
    baseline_metrics = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        metrics = evaluate_regression(y_test, y_pred, name)
        baseline_metrics.append({"model": name, **metrics})
        joblib.dump(model, f"baseline_{name}.joblib")

    pd.DataFrame(baseline_metrics).to_csv("baseline_metrics.csv", index=False)
    print("\nAll baseline models trained and saved.")
    print("Summary written to baseline_metrics.csv")


if __name__ == "__main__":
    train_baselines()
