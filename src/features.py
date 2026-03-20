"""
features.py
-----------
Domain-specific descriptor extraction (ASE / pymatgen) and preprocessing pipeline.

Run this script directly to:
  1. Extract atomic descriptors from the 'atoms' column (if available).
  2. Build the preprocessing pipeline (imputation + scaling).
  3. Perform 5-fold cross-validation on a baseline Ridge model.
  4. Save the processed train/test split to split_data.joblib.

Usage
-----
    python src/features.py
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

from config import (
    ASE_AVAILABLE, PYM_AVAILABLE,
    DATA_PATH, RANDOM_STATE, CV_FOLDS,
    crossval_report,
)

# Optional ASE import (already attempted in config, import the symbol here if available)
if ASE_AVAILABLE:
    from ase.data import atomic_masses


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate atomic-level descriptors from the 'atoms' column.

    If neither ASE nor pymatgen is installed the function returns the
    dataframe unchanged (minus the 'atoms' column if present).

    Descriptors produced (when possible)
    -------------------------------------
    n_atoms    : total number of atoms in the unit cell
    n_species  : number of distinct chemical species
    mean_Z     : mean atomic number
    std_Z      : standard deviation of atomic numbers
    mean_mass  : mean atomic mass (amu)

    Parameters
    ----------
    df : DataFrame  Raw dataframe potentially containing an 'atoms' column.

    Returns
    -------
    DataFrame with 'atoms' replaced by numeric descriptors.
    """
    if "atoms" not in df.columns:
        return df

    rows = []
    for val in df["atoms"]:
        desc = {}

        if ASE_AVAILABLE:
            try:
                Zs     = val.get_atomic_numbers()
                masses = [atomic_masses[z] for z in Zs]
                desc   = {
                    "n_atoms":   len(Zs),
                    "n_species": len(set(Zs)),
                    "mean_Z":    np.mean(Zs),
                    "std_Z":     np.std(Zs),
                    "mean_mass": np.mean(masses),
                }
            except Exception:
                desc = {"n_atoms": np.nan}

        elif PYM_AVAILABLE:
            try:
                species = [site.specie for site in val]
                Zs      = [s.Z for s in species]
                masses  = [float(s.atomic_mass) for s in species]
                desc    = {
                    "n_atoms":   len(Zs),
                    "n_species": len(set(Zs)),
                    "mean_Z":    np.mean(Zs),
                    "std_Z":     np.std(Zs),
                    "mean_mass": np.mean(masses),
                }
            except Exception:
                desc = {"n_atoms": np.nan}

        else:
            desc = {"n_atoms": np.nan}

        rows.append(desc)

    return pd.concat(
        [df.drop(columns=["atoms"]), pd.DataFrame(rows, index=df.index)],
        axis=1,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_and_save_split() -> None:
    """Load data, extract features, build preprocessor, run CV, save split."""

    # --- Load and augment ---
    df = pd.read_pickle(DATA_PATH)
    df = extract_descriptors(df)
    df = df.dropna(subset=["energy_per_atom"]).reset_index(drop=True)

    X = df.drop(columns=["energy_per_atom"]).select_dtypes(include=[np.number])
    y = df["energy_per_atom"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    numeric_cols = X_train.columns.tolist()

    # --- Preprocessing: impute missing values, then standardize ---
    preproc = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler",  StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        ],
        remainder="drop",
    )

    # --- Baseline CV check with Ridge ---
    baseline_pipe = Pipeline(
        [("preproc", preproc), ("model", Ridge(random_state=RANDOM_STATE))]
    )
    cv_result = crossval_report(baseline_pipe, X_train, y_train)
    print("Ridge CV →", cv_result)

    # --- Persist split and preprocessor for downstream scripts ---
    joblib.dump(
        {
            "X_train":      X_train,
            "X_test":       X_test,
            "y_train":      y_train,
            "y_test":       y_test,
            "numeric_cols": numeric_cols,
            "preproc":      preproc,
        },
        "split_data.joblib",
    )
    print("Saved: split_data.joblib")


if __name__ == "__main__":
    build_and_save_split()
