"""
train_tuned.py
--------------
Hyperparameter optimisation for Random Forest, Gradient Boosting, and SVR
using RandomizedSearchCV with 5-fold cross-validation (scored on MAE).

Each search explores 10 random parameter combinations per model.
Best estimators are saved as  best_rf.joblib / best_gb.joblib / best_svr.joblib.
A summary CSV is written to   tuned_models_summary.csv.

Usage
-----
    python src/train_tuned.py

Prerequisite
------------
    Run features.py first — this script loads split_data.joblib.
"""

import pandas as pd
import joblib

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

from config import RANDOM_STATE, CV_FOLDS, N_JOBS, evaluate_regression


def tune_models() -> None:
    # --- Load preprocessed split ---
    data        = joblib.load("split_data.joblib")
    X_train     = data["X_train"]
    X_test      = data["X_test"]
    y_train     = data["y_train"]
    y_test      = data["y_test"]
    preproc     = data["preproc"]

    search_kwargs = dict(
        n_iter=10,
        cv=CV_FOLDS,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
    )

    tuned_metrics = []

    # ------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------
    rf_params = {
        "model__n_estimators":    [100, 200, 300, 400],
        "model__max_depth":       [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf":  [1, 2, 4],
    }
    rf_pipe = Pipeline(
        [("preproc", preproc),
         ("model",   RandomForestRegressor(random_state=RANDOM_STATE))]
    )
    rf_search = RandomizedSearchCV(rf_pipe, rf_params, **search_kwargs)
    rf_search.fit(X_train, y_train)
    print("Best RF params:", rf_search.best_params_)
    joblib.dump(rf_search.best_estimator_, "best_rf.joblib")
    rf_metrics = evaluate_regression(
        y_test, rf_search.best_estimator_.predict(X_test), "Random Forest (Tuned)"
    )
    tuned_metrics.append({"model": "RandomForest", **rf_metrics})

    # ------------------------------------------------------------------
    # Gradient Boosting
    # ------------------------------------------------------------------
    gb_params = {
        "model__n_estimators":  [100, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth":     [3, 5, 7],
        "model__subsample":     [0.7, 0.85, 1.0],
    }
    gb_pipe = Pipeline(
        [("preproc", preproc),
         ("model",   GradientBoostingRegressor(random_state=RANDOM_STATE))]
    )
    gb_search = RandomizedSearchCV(gb_pipe, gb_params, **search_kwargs)
    gb_search.fit(X_train, y_train)
    print("Best GB params:", gb_search.best_params_)
    joblib.dump(gb_search.best_estimator_, "best_gb.joblib")
    gb_metrics = evaluate_regression(
        y_test, gb_search.best_estimator_.predict(X_test), "Gradient Boosting (Tuned)"
    )
    tuned_metrics.append({"model": "GradientBoosting", **gb_metrics})

    # ------------------------------------------------------------------
    # SVR  (scaling already handled by the preprocessor in the pipeline)
    # ------------------------------------------------------------------
    svr_params = {
        "model__C":       [0.1, 1, 10, 100],
        "model__gamma":   ["scale", "auto", 0.01, 0.1],
        "model__epsilon": [0.01, 0.1, 0.2],
    }
    svr_pipe = Pipeline(
        [("preproc", preproc),
         ("model",   SVR())]
    )
    svr_search = RandomizedSearchCV(svr_pipe, svr_params, **search_kwargs)
    svr_search.fit(X_train, y_train)
    print("Best SVR params:", svr_search.best_params_)
    joblib.dump(svr_search.best_estimator_, "best_svr.joblib")
    svr_metrics = evaluate_regression(
        y_test, svr_search.best_estimator_.predict(X_test), "SVR (Tuned)"
    )
    tuned_metrics.append({"model": "SVR", **svr_metrics})

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    pd.DataFrame(tuned_metrics).to_csv("tuned_models_summary.csv", index=False)
    print("\nHyperparameter tuning complete.")
    print("Summary written to tuned_models_summary.csv")


if __name__ == "__main__":
    tune_models()
