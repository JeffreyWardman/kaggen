import importlib
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataloader import DataLoader
from warmup_scheduler import GradualWarmupScheduler

from ..dataset import Collator
from ..models import PremadeModel
from .optimizers import Over9000


def get_train_val_datasets(df: pd.DataFrame,
                           path_column: str,
                           target_column: str,
                           fold_idx: int,
                           num_folds: int = 5,
                           seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    df['fold'] = -1
    no_duplicates = df.drop_duplicates()
    for val_fold_idx, (train_idx,
                       valid_idx) in enumerate(skf.split(no_duplicates[path_column], no_duplicates[target_column])):
        df.loc[df.index.isin(valid_idx), 'fold'] = val_fold_idx

    train_idx = np.where((df['fold'] != fold_idx))[0]
    valid_idx = np.where((df['fold'] == fold_idx))[0]

    df_train = df.loc[train_idx]
    df_valid = df.loc[valid_idx]

    # Check no data has missing fold IDs.
    assert np.sum(df_train['fold'] == -1) == 0
    assert np.sum(df_valid['fold'] == -1) == 0

    return df_train, df_valid


def csvs_to_dataloaders(df_train: pd.DataFrame,
                        df_valid: pd.DataFrame,
                        dataset_name: str,
                        path_column: str,
                        target_column: str,
                        classes: Dict[str, int],
                        transform: Any,
                        fold_idx: int,
                        num_folds: int,
                        seed: int,
                        image_shape: Tuple[int, int, int] = (128, 512, 3),
                        batch_size: int = 128,
                        num_workers: int = 0,
                        percentile: float = 100) -> Tuple[DataLoader, DataLoader]:
    """Generates training and validation dataset loaders.

    Arguments:
        df_train {pd.DataFrame} -- Training set dataframe
        df_valid {pd.DataFrame} -- Validation set dataframe
        dataset_name {str} -- Name of dataset to import from kaggen.train.datasets.
        path_column {str} -- Column name containing data paths
        target_column {str} -- Column name containing target values
        classes {Dict[str, int]} -- Class names and index as keys and values respectively
        transform {Any} -- Transformations
    Keyword Arguments:
        image_shape {Tuple[int, int, int]} -- Image shape (H, W, C) (default: {(128, 512, 3)})
        batch_size {int} -- Batch size (default: {128})
        num_workers {int} -- Number of workers (default: {0})
        percentile {float} -- Percentile for collator (default: {100})

    Returns:
        Tuple[DataLoader, DataLoader] -- Train and validation data loaders (respectively)
    """
    df_train, df_valid = get_train_val_datasets(df_train,
                                                path_column=path_column,
                                                target_column=target_column,
                                                fold_idx=fold_idx,
                                                num_folds=num_folds,
                                                seed=seed)

    # TODO test
    dataset = getattr(importlib.import_module('kaggen.train.datasets'), dataset_name)
    dataset_args = {
        'path_column': path_column,
        'target_column': target_column,
        'classes': classes,
    }
    train_dataset_args = dataset_args.copy()
    train_dataset_args.update({'df': df_train, 'transform': transform})
    valid_dataset_args = dataset_args.copy()
    valid_dataset_args.update({'df': df_valid, 'transform': None})

    train_set = dataset(**train_dataset_args)
    valid_set = dataset(**valid_dataset_args)

    train_collate = Collator(percentile)
    valid_collate = Collator(percentile)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=train_collate,
                              pin_memory=True,
                              num_workers=num_workers)
    valid_loader = DataLoader(valid_set,
                              batch_size=batch_size,
                              collate_fn=valid_collate,
                              pin_memory=True,
                              num_workers=num_workers)

    return train_loader, valid_loader


def get_dataloader(module: str, func_name: str, parameters: Dict[str, Any], **kwargs) -> Any:
    return getattr(importlib.import_module(module), func_name)(**kwargs, **parameters)


def get_loss_function(loss_module: str, loss_function: str, **kwargs) -> Any:
    """Get loss function from module

    Arguments:
        loss_module {str} -- Module where loss function is collected from
        loss_function {str} -- Name of loss function

    Returns:
        Any -- Loss function
    """
    return getattr(importlib.import_module(loss_module), loss_function)(**kwargs)


def get_lr_scheduler(scheduler_module: str,
                     scheduler_name: str,
                     optimizer: Any,
                     num_epochs: int,
                     warmup_epoch: int = 0,
                     warmup_factor: float = 0.):
    """Generates scheduler with gradual warmup.

    Arguments:
        scheduler_module {str} -- Name of module which scheduler is collected from
        scheduler_name {str} -- Name of scheduler
        optimizer {[type]} -- Optimizer  # TODO
        num_epochs {int} -- Number of epochs
        warmup_epoch {int} -- Epoch at which warmup ends
        warmup_factor {float} -- Warmup factor (default: {0})

    Returns:
        GradualWarmupScheduper -- Warmup schduler
    """
    scheduler = getattr(importlib.import_module(scheduler_module), scheduler_name)(optimizer, num_epochs - warmup_epoch)
    if warmup_epoch:
        scheduler = GradualWarmupScheduler(optimizer,
                                           multiplier=warmup_factor,
                                           total_epoch=warmup_epoch,
                                           after_scheduler=scheduler)
    return scheduler


def get_model(model_config, num_classes: int):
    """Gets model

    Arguments:
        model_name {str} -- Model type (e.g. resnet18)
        module {str} -- Module where model backbone can be collected from
        num_classes {int} -- Number of classes for output layer

    Returns:
        models.PremadeModel -- Model
    """
    return getattr(importlib.import_module('kaggen.models'), model_config['class_name'])(num_classes=num_classes,
                                                                                         **model_config['parameters'])


def get_optimizer(optimizer_module: str, optimizer_name: str, model, lr: float, weight_decay: float = 1e-3):
    """Gets optimizer

    Arguments:
        optimizer_module {str} -- Module where optimizer is collected from
        optimizer_name {str} -- Optimizer name
        model {[type]} -- PyTorch model  # TODO
        lr {float} -- Learning rate

    Keyword Arguments:
        weight_decay {float} -- Weight decay (default: {1e-3})

    Returns:
        [type] -- Optimizer  # TODO
    """
    func = getattr(importlib.import_module(optimizer_module), optimizer_name)
    optimizer = func(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer
