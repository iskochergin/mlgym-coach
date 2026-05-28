from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
HIDDEN_LABELS_DIR = ROOT / "hidden_labels"
RANDOM_STATE = 20260527
TEST_SIZE = 0.25


def _clean_columns(columns: list[str]) -> list[str]:
    return [
        column.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        for column in columns
    ]


def _write_split(
    task_id: str,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    stratify: pd.Series | None = None,
) -> None:
    task_data_dir = DATA_DIR / task_id
    task_hidden_dir = HIDDEN_LABELS_DIR / task_id
    task_data_dir.mkdir(parents=True, exist_ok=True)
    task_hidden_dir.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    train = x_train.reset_index(drop=True).copy()
    train["target"] = y_train.reset_index(drop=True)

    train.to_csv(task_data_dir / "train.csv", index=False)
    x_test.reset_index(drop=True).to_csv(task_data_dir / "test_features.csv", index=False)
    y_test.reset_index(drop=True).to_frame("target").to_csv(
        task_hidden_dir / "y_test.csv",
        index=False,
    )


def prepare_breast_cancer() -> None:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    features.columns = _clean_columns(list(features.columns))
    target = dataset.target.rename("target")

    _write_split(
        "breast_cancer_roc_auc",
        features,
        target,
        stratify=target,
    )


def prepare_diabetes() -> None:
    dataset = load_diabetes(as_frame=True)
    features = dataset.data.copy()
    features.columns = _clean_columns(list(features.columns))
    target = dataset.target.rename("target")

    _write_split("diabetes_rmse", features, target)


def main() -> None:
    prepare_breast_cancer()
    prepare_diabetes()


if __name__ == "__main__":
    main()
