"""
run_pipeline.py
---------------
End-to-end pipeline runner.  Executes all four stages in order:

    1. Feature engineering + preprocessing  (features.py)
    2. Baseline models                       (train_baseline.py)
    3. Hyperparameter tuning                 (train_tuned.py)
    4. Stacking ensemble + interpretability  (train_ensemble.py)
    5. Full visualisation + metric report    (visualize.py)

Usage
-----
    python run_pipeline.py

All intermediate artefacts (.joblib, .csv) and output plots (.png)
are written to the working directory.
"""

from src.features        import build_and_save_split
from src.train_baseline  import train_baselines
from src.train_tuned     import tune_models
from src.train_ensemble  import train_ensemble
from src.visualize       import run_visualizations


def main() -> None:
    print("=" * 60)
    print("STEP 1 — Feature engineering & preprocessing")
    print("=" * 60)
    build_and_save_split()

    print("\n" + "=" * 60)
    print("STEP 2 — Baseline models")
    print("=" * 60)
    train_baselines()

    print("\n" + "=" * 60)
    print("STEP 3 — Hyperparameter tuning")
    print("=" * 60)
    tune_models()

    print("\n" + "=" * 60)
    print("STEP 4 — Stacking ensemble & interpretability")
    print("=" * 60)
    train_ensemble()

    print("\n" + "=" * 60)
    print("STEP 5 — Visualisation & final report")
    print("=" * 60)
    run_visualizations()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
