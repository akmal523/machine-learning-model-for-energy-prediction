# ML Energy Prediction

A machine learning pipeline for predicting **energy per atom** (eV/atom) of materials from structural and compositional descriptors.  
The pipeline covers data loading, feature engineering, baseline training, hyperparameter tuning, stacking ensemble, SHAP interpretability, and full visualisation.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Results

| Model                        | MAE   | RMSE  | R²    |
|------------------------------|-------|-------|-------|
| **Stacking Ensemble**        | 0.136 | 0.239 | 0.918 |
| Gradient Boosting (Tuned)    | 0.146 | 0.243 | 0.915 |
| Random Forest (Tuned)        | 0.144 | 0.259 | 0.903 |
| Random Forest (Baseline)     | 0.144 | 0.259 | 0.903 |
| Gradient Boosting (Baseline) | 0.179 | 0.284 | 0.883 |
| SVR (Tuned)                  | 0.153 | 0.356 | 0.817 |
| SVR (Baseline)               | 0.168 | 0.371 | 0.801 |
| Linear Regression (Baseline) | 0.257 | 0.435 | 0.727 |

Detailed explanation of every number — what the metrics mean, why each model performs as it does, and where the ceiling is — lives in [`results/results.txt`](results/results.txt).

---

## Project Structure

```
machine-learning-model-for-energy-prediction/
│
├── src/
│   ├── config.py            # Constants, shared imports, helper functions
│   ├── features.py          # Domain descriptor extraction + preprocessing pipeline
│   ├── train_baseline.py    # Baseline models (LR, RF, GB, SVR)
│   ├── train_tuned.py       # Hyperparameter tuning via RandomizedSearchCV
│   ├── train_ensemble.py    # Stacking ensemble + feature importances + SHAP
│   └── visualize.py         # All plots + final metrics comparison table
│
├── results/
│   └── results.txt          # Full analysis of model results and metrics
│
├── run_pipeline.py          # Single entry point — runs all 5 stages in order
├── dataset1.pckl.gz         # Materials dataset (pickled DataFrame)
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Pipeline

```
dataset1.pckl.gz
       │
       ▼
 features.py  ─── Descriptor extraction (ASE / pymatgen, optional)
               ── Preprocessing: SimpleImputer → StandardScaler
               ── 5-fold CV on Ridge (sanity check)
               ── Saves split_data.joblib
       │
       ├──► train_baseline.py  ── LR, RF, GB, SVR (default params)
       │                        ── Saves baseline_<Model>.joblib
       │
       ├──► train_tuned.py     ── RandomizedSearchCV (n_iter=10, 5-fold)
       │                        ── RF, GB, SVR
       │                        ── Saves best_rf / best_gb / best_svr .joblib
       │
       ├──► train_ensemble.py  ── StackingRegressor (RF + GB + SVR → Ridge)
       │                        ── Feature importances (RF, GB)
       │                        ── SHAP global + local (optional)
       │                        ── Saves final_stacking_model.joblib
       │
       └──► visualize.py       ── Actual vs Predicted, Residuals (all models)
                                ── MAE / RMSE / R² comparison bar chart
                                ── Feature importance plots
                                ── SHAP summary + waterfall (optional)
```

---

## Installation

```bash
git clone https://github.com/akmal523/machine-learning-model-for-energy-prediction.git
cd machine-learning-model-for-energy-prediction
pip install -r requirements.txt
```

ASE, pymatgen, and shap are **optional**. If not installed, the pipeline skips domain-specific descriptors and SHAP analysis and runs on numeric columns only.

---

## Usage

**Run the full pipeline in one command:**

```bash
python run_pipeline.py
```

**Or run each stage individually:**

```bash
python src/features.py          # step 1 — must run first
python src/train_baseline.py    # step 2
python src/train_tuned.py       # step 3
python src/train_ensemble.py    # step 4
python src/visualize.py         # step 5
```

Each script is self-contained and documented. Intermediate artefacts (`.joblib`, `.csv`) are written to the working directory and reused by downstream scripts.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy`, `pandas` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `scikit-learn` | Pipelines, models, tuning |
| `joblib` | Model persistence |
| `shap` | Feature interpretability *(optional)* |
| `ase` | Atomic descriptor extraction *(optional)* |
| `pymatgen` | Crystal structure features *(optional)* |
| `matminer` | Materials-specific featurisation *(optional)* |

---

## License

[MIT](LICENSE)
