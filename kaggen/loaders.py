import importlib
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def get_train_val_datasets(
    df: pd.DataFrame, path_column: str, target_column: str, fold_idx: int, num_folds: int = 5, seed: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Gets train and validation datasets from dataframe by splitting it based on num_folds.

    Arguments:
        df {pd.DataFrame} -- Train+validation dataset
        path_column {str} -- Column in dataframe with data locations.
        target_column {str} -- Classification column
        fold_idx {int} -- Fold index

    Keyword Arguments:
        num_folds {int} -- Number of folds (default: {5})
        seed {int} -- Seed to control reproducibility (default: {0})

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame] -- Train and validation datasets (respectively)
    """
    skf = StratifiedKFold(num_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    no_duplicates = df.drop_duplicates()
    for val_fold_idx, (train_idx, valid_idx) in enumerate(
        skf.split(no_duplicates[path_column], no_duplicates[target_column])
    ):
        df.loc[df.index.isin(valid_idx), "fold"] = val_fold_idx

    train_idx = np.where((df["fold"] != fold_idx))[0]
    valid_idx = np.where((df["fold"] == fold_idx))[0]

    df_train = df.loc[train_idx]
    df_valid = df.loc[valid_idx]

    # Check no data has missing fold IDs.
    assert np.sum(df_train["fold"] == -1) == 0
    assert np.sum(df_valid["fold"] == -1) == 0

    return df_train, df_valid
