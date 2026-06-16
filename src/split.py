"""Train/test splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src import SEED
from src.schema import TARGET_COLUMN


@dataclass(frozen=True)
class DataSplit:
    """Container for a stratified train/test split."""

    train: pd.DataFrame
    test: pd.DataFrame
    summary: pd.DataFrame


def build_split_summary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Summarize split sizes and target distribution."""
    rows = []
    for split_name, frame in {"train": train, "test": test}.items():
        counts = frame[TARGET_COLUMN].value_counts().sort_index()
        rows.append(
            {
                "split": split_name,
                "rows": len(frame),
                "target_0_count": int(counts.get(0, 0)),
                "target_1_count": int(counts.get(1, 0)),
                "target_1_percentage": round(float(counts.get(1, 0) / len(frame) * 100), 2),
            }
        )
    return pd.DataFrame(rows)


def stratified_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = SEED,
) -> DataSplit:
    """Create the single held-out split used throughout the project."""
    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET_COLUMN],
        random_state=random_state,
    )
    train = train.sort_index().reset_index(drop=True)
    test = test.sort_index().reset_index(drop=True)
    summary = build_split_summary(train, test)
    return DataSplit(train=train, test=test, summary=summary)
