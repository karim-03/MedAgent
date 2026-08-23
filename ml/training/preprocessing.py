"""
preprocessing.py
Milestone P1 (dataset audit + preprocessing) for MedAgent.

Loads the raw Heart Disease (Cleveland) dataset, applies documented cleaning
decisions, and produces a reproducible train/test split ready for model
training in the next milestone.

Every cleaning decision made here is backed by the audit findings recorded
in docs/data_audit_findings.md — nothing here is silent or arbitrary.
"""

from pathlib import Path
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw/heart_raw.csv")
PROCESSED_DIR = Path("data/processed")

# Columns whose values fall outside the documented UCI codebook range and are
# therefore treated as disguised missing values (see audit findings).
INVALID_CA_VALUE = 4        # valid range per codebook: 0-3
INVALID_THAL_VALUE = 0      # valid range per codebook: 1-3

# Nominal (unordered) categorical columns -> require one-hot encoding
# downstream, must NOT be treated as ordinal/continuous by linear models.
NOMINAL_COLUMNS = ["cp", "restecg", "slope", "thal"]

# True binary columns -> safe to use as-is (0/1).
BINARY_COLUMNS = ["sex", "fbs", "exang"]

# Continuous numeric columns -> candidates for scaling (Logistic Regression
# pipeline only; tree-based models do not need scaling).
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

TARGET_COLUMN = "target"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV exactly as downloaded, no cleaning applied yet."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("Loaded raw data: %s rows, %s columns", *df.shape)
    return df


def clean_data(df: pd.DataFrame, flip_target: bool = True) -> pd.DataFrame:
    """
    Apply documented cleaning steps:

    1. Drop exact duplicate rows.
    2. Drop rows with out-of-codebook placeholder values in `ca`/`thal`
       (these are disguised missing values, not real categories).
    3. Optionally flip target polarity.

       Audit finding: correlation direction of exang, oldpeak, ca, and
       thalach against `target` in this specific CSV mirror is consistently
       the OPPOSITE of what clinical literature predicts for "1 = disease
       present" (e.g. higher oldpeak / more vessels blocked / exercise
       angina all point toward target == 0, not target == 1). This matches
       a documented community discussion around this dataset mirror
       (Kaggle: ronitf/heart-disease-uci, discussion #105877) reporting the
       same inversion relative to the original UCI codebook description.

       flip_target=True (default) treats target==1 as "heart disease
       present" for the rest of this project, by flipping the raw column.
       Set to False if you have independently confirmed the raw polarity
       is correct and want to override this decision.
    """
    before = len(df)
    df = df.drop_duplicates().copy()
    logger.info("Dropped %d duplicate row(s)", before - len(df))

    before = len(df)
    mask_invalid = (df["ca"] == INVALID_CA_VALUE) | (df["thal"] == INVALID_THAL_VALUE)
    df = df.loc[~mask_invalid].copy()
    logger.info(
        "Dropped %d row(s) with out-of-codebook ca/thal placeholder values",
        before - len(df),
    )

    if flip_target:
        df[TARGET_COLUMN] = 1 - df[TARGET_COLUMN]
        logger.info("Flipped target polarity: 1 now means disease PRESENT")

    df = df.reset_index(drop=True)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified train/test split — stratified because we want the disease
    prevalence preserved in both splits, not just row count."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Split: train=%d (pos rate=%.3f), test=%d (pos rate=%.3f)",
        len(X_train), y_train.mean(), len(X_test), y_test.mean(),
    )
    return X_train, X_test, y_train, y_test


def run_pipeline(save: bool = True):
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    X_train, X_test, y_train, y_test = split_data(df_clean)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        X_train.assign(target=y_train).to_csv(PROCESSED_DIR / "train.csv", index=False)
        X_test.assign(target=y_test).to_csv(PROCESSED_DIR / "test.csv", index=False)
        logger.info("Saved processed train/test splits to %s", PROCESSED_DIR)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_pipeline()
