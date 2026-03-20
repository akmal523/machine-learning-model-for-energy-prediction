"""
visualize.py
------------
Full evaluation and visualisation pass across all trained models.

Produces
--------
  - Actual vs Predicted scatter plots (per model)
  - Residual plots (per model)
  - Side-by-side MAE / RMSE / R² bar chart (all models)
  - Feature importance bar charts (RF, GB)
  - SHAP summary + waterfall plots (RF, GB — if shap installed)
  - Metrics table printed to stdout

Usage
-----
    python src/visualize.py

Prerequisites
-------------
    Run the full pipeline first:
        python src/features.py
        python src/train_baseline.py
        python src/train_tuned.py
        python src/train_ensemble.py
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import SHAP_AVAILABLE, numeric_cols_from_data

if SHAP_AVAILABLE:
    import shap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred) -> tuple:
    """Return (MAE, RMSE, R²)."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2


def plot_model_performance(y_true, y_pred, model_name: str) -> None:
    """Scatter (actual vs predicted) + residual plot for one model."""
    # Actual vs Predicted
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", lw=2, label="Perfect prediction")
    plt.xlabel("Actual Energy per Atom (eV/atom)")
    plt.ylabel("Predicted Energy per Atom (eV/atom)")
    plt.title(f"Actual vs Predicted — {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plot_actual_vs_pred_{model_name.replace(' ', '_').lower()}.png", dpi=150)
    plt.show()

    # Residuals
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color="r", linestyle="--", lw=2, label="Zero residual")
    plt.xlabel("Predicted Energy per Atom (eV/atom)")
    plt.ylabel("Residuals")
    plt.title(f"Residuals — {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plot_residuals_{model_name.replace(' ', '_').lower()}.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_visualizations() -> None:
    # --- Load split ---
    data         = joblib.load("split_data.joblib")
    X_test_full  = data["X_test"]
    y_test       = data["y_test"]
    numeric_cols = data["numeric_cols"]

    # Baseline models were trained on features without domain descriptors
    baseline_features = [
        c for c in X_test_full.columns
        if c not in ["n_atoms", "n_species", "mean_Z", "std_Z", "mean_mass"]
    ]
    X_test_baseline = X_test_full[baseline_features]

    # --- Load all models ---
    baseline_models = {
        "Linear Regression (Baseline)":  joblib.load("baseline_LinearRegression.joblib"),
        "Random Forest (Baseline)":       joblib.load("baseline_RandomForest.joblib"),
        "Gradient Boosting (Baseline)":   joblib.load("baseline_GradientBoosting.joblib"),
        "SVR (Baseline)":                 joblib.load("baseline_SVR.joblib"),
    }
    tuned_models = {
        "Random Forest (Tuned)":      joblib.load("best_rf.joblib"),
        "Gradient Boosting (Tuned)":  joblib.load("best_gb.joblib"),
        "SVR (Tuned)":                joblib.load("best_svr.joblib"),
    }
    stack_pipe = joblib.load("final_stacking_model.joblib")

    # --- Evaluate all models ---
    metrics_list = []

    for name, model in baseline_models.items():
        y_pred = model.predict(X_test_baseline)
        plot_model_performance(y_test, y_pred, name)
        mae, rmse, r2 = compute_metrics(y_test, y_pred)
        metrics_list.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

    for name, model in tuned_models.items():
        y_pred = model.predict(X_test_full)
        plot_model_performance(y_test, y_pred, name)
        mae, rmse, r2 = compute_metrics(y_test, y_pred)
        metrics_list.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

    y_pred_stack = stack_pipe.predict(X_test_full)
    plot_model_performance(y_test, y_pred_stack, "Stacking Ensemble")
    mae, rmse, r2 = compute_metrics(y_test, y_pred_stack)
    metrics_list.append({"Model": "Stacking Ensemble", "MAE": mae, "RMSE": rmse, "R2": r2})

    # --- Metrics table ---
    metrics_df = pd.DataFrame(metrics_list).sort_values("R2", ascending=False)
    print("\n" + metrics_df.to_string(index=False))

    # --- Comparison bar chart ---
    metrics_long = metrics_df.melt(
        id_vars="Model", value_vars=["MAE", "RMSE", "R2"],
        var_name="Metric", value_name="Value"
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=metrics_long, x="Model", y="Value", hue="Metric")
    plt.xticks(rotation=45, ha="right")
    plt.title("Model Comparison — MAE / RMSE / R²")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig("plot_model_comparison.png", dpi=150)
    plt.show()

    # --- Feature importances ---
    for label, model in [("Random Forest", tuned_models["Random Forest (Tuned)"]),
                          ("Gradient Boosting", tuned_models["Gradient Boosting (Tuned)"])]:
        estimator = model.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            imp_df = (
                pd.DataFrame({"feature": numeric_cols, "importance": estimator.feature_importances_})
                .sort_values("importance", ascending=False)
            )
            plt.figure(figsize=(10, 6))
            sns.barplot(x="importance", y="feature", data=imp_df)
            plt.title(f"Feature Importance — {label}")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"plot_importance_{label.replace(' ', '_').lower()}.png", dpi=150)
            plt.show()

    # --- SHAP ---
    if SHAP_AVAILABLE:
        sample_size = min(200, X_test_full.shape[0])
        for label, model in [("Random Forest", tuned_models["Random Forest (Tuned)"]),
                              ("Gradient Boosting", tuned_models["Gradient Boosting (Tuned)"])]:
            estimator = model.named_steps["model"]
            preproc   = model.named_steps["preproc"]
            X_trans   = preproc.transform(X_test_full.iloc[:sample_size])
            explainer   = shap.TreeExplainer(estimator)
            shap_values = explainer(X_trans)
            print(f"\nSHAP — {label}")
            shap.summary_plot(shap_values, X_test_full.iloc[:sample_size], feature_names=numeric_cols)
            shap.plots.waterfall(shap_values[0])
    else:
        print("\nshap not installed — SHAP plots skipped.")


if __name__ == "__main__":
    run_visualizations()
