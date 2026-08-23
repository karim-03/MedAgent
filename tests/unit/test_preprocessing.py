"""
Unit tests for ml/training/preprocessing.py
Run with: pytest tests/unit/test_preprocessing.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.training.preprocessing import (
    load_raw_data,
    clean_data,
    split_data,
    TARGET_COLUMN,
)


@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data()


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean_data(raw_df)


def test_raw_data_loads_with_expected_shape(raw_df):
    # 303 rows, 14 columns is the known shape of this dataset mirror.
    assert raw_df.shape == (303, 14)


def test_clean_data_removes_duplicates(clean_df):
    assert clean_df.duplicated().sum() == 0


def test_clean_data_removes_invalid_placeholder_rows(clean_df):
    assert (clean_df["ca"] == 4).sum() == 0
    assert (clean_df["thal"] == 0).sum() == 0


def test_target_is_binary(clean_df):
    assert set(clean_df[TARGET_COLUMN].unique()) == {0, 1}


def test_target_flip_inverts_polarity(raw_df):
    unflipped = clean_data(raw_df, flip_target=False)
    flipped = clean_data(raw_df, flip_target=True)
    # Same rows survive cleaning either way; only the label should differ.
    assert len(unflipped) == len(flipped)
    assert (unflipped[TARGET_COLUMN] + flipped[TARGET_COLUMN] == 1).all()


def test_split_is_stratified_within_tolerance(clean_df):
    X_train, X_test, y_train, y_test = split_data(clean_df, test_size=0.2, random_state=42)
    overall_rate = clean_df[TARGET_COLUMN].mean()
    assert abs(y_train.mean() - overall_rate) < 0.05
    assert abs(y_test.mean() - overall_rate) < 0.05


def test_split_has_no_row_overlap(clean_df):
    X_train, X_test, _, _ = split_data(clean_df, test_size=0.2, random_state=42)
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_split_reproducible_with_fixed_seed(clean_df):
    a = split_data(clean_df, random_state=42)
    b = split_data(clean_df, random_state=42)
    pd.testing.assert_frame_equal(a[0], b[0])
