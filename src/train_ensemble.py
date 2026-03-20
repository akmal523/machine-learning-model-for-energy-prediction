"""
train_ensemble.py
-----------------
Build a stacking ensemble from the three tuned base learners
(Random Forest, Gradient Boosting, SVR) with a Ridge meta-model.

Additionally:
  - Extracts tree-based feature importances for RF and GB.
  - Computes SHAP values for global and local interpretability (if shap is installed).

Artifacts saved
---------------
  final_stacking_model.joblib  — the full trained stacking pipeline.

Usage
-----
    python src/train_ensemble.py

Prerequisites
-------------
    Run features.py then train_tuned.py first.
    Requires: split_data.joblib, best_rf.joblib, best_gb.joblib, best_svr.joblib
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

from config import SHAP_AVAILABLE, evaluate_regression

if SHAP_AVAILABLE:
    import shap


def train_ensemble() -> None:
    # --- Load data and tuned base learners ---
    data        = joblib.load("split_data.joblib")
    X_train     = data["X_train"]
    X_test      = data["X_test"]
    y_train     = data["y_train"]
    y_test      = data["y_test"]
    numeric_cols = data["numeric_cols"]

    best_rf  = joblib.load("best_rf.joblib")
    best_gb  = joblib.load("best_gb.joblib")
    best_svr = joblib.load("best_svr.joblib")

    # ------------------------------------------------------------------
    # Stacking ensemble
    # Each base learner exposes its own preprocessing pipeline.
    # The Ridge meta-model learns to combine their out-of-fold predictions.
    # passthrough=True also feeds the original features to the meta-model,
    # giving it access to raw signal not captured by the base learners.
    # ------------------------------------------------------------------
    stack_pipe = StackingRegressor(
        estimators=[
            ("rf",  best_rf),
            ("gb",  best_gb),
            ("svr", best_svr),
        ],
        final_estimator=Ridge(),
        n_jobs=-1,
        passthrough=True,
    )
    stack_pipe.fit(X_train, y_train)

    stack_preds  = stack_pipe.predict(X_test)
    stack_metrics = evaluate_regression(y_test, stack_preds, "Stacking Ensemble")
    print(stack_metrics)

    # ------------------------------------------------------------------
    # Feature importances — tree models only
    # ------------------------------------------------------------------
    for label, model in [("Random Forest", best_rf), ("Gradient Boosting", best_gb)]:
        estimator = model.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            imp_df = (
                pd.DataFrame(
                    {"feature": numeric_cols, "importance": estimator.feature_importances_}
                )
                .sort_values("importance", ascending=False)
            )
            plt.figure(figsize=(12, 6))
            sns.barplot(x="importance", y="feature", data=imp_df)
            plt.title(f"Feature Importance — {label}")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"feature_importance_{label.replace(' ', '_').lower()}.png", dpi=150)
            plt.show()
            print(f"Feature importance plot saved for {label}.")

    # ------------------------------------------------------------------
    # SHAP interpretability (optional)
    # ------------------------------------------------------------------
    if SHAP_AVAILABLE:
        sample_size = min(200, X_test.shape[0])

        for label, model in [("Random Forest", best_rf), ("Gradient Boosting", best_gb)]:
            estimator = model.named_steps["model"]
            preproc   = model.named_steps["preproc"]
            X_trans   = preproc.transform(X_test.iloc[:sample_size])

            explainer   = shap.TreeExplainer(estimator)
            shap_values = explainer(X_trans)

            print(f"\nSHAP summary — {label}")
            shap.summary_plot(shap_values, X_test.iloc[:sample_size], feature_names=numeric_cols)
            shap.plots.waterfall(shap_values[0])
    else:
        print("\nshap not installed — skipping SHAP analysis.")
        print("Install with:  pip install shap")

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    joblib.dump(stack_pipe, "final_stacking_model.joblib")
    print("\nSaved: final_stacking_model.joblib")


if __name__ == "__main__":
    train_ensemble()
