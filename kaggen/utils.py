import logging
import os
import random
from typing import Any, Dict, Union

import numpy as np
import torch
import wandb
import yaml


def set_seed(seed: int = 0) -> None:
    """Sets random seed

    Keyword Arguments:
        seed {int} -- Seed value (default: {0})
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_config(config: Union[str, Dict], inheritance_key: str = 'INHERIT') -> Dict[str, Any]:
    """Reads YAML configuration file with nested inheritance from other YAML files.

    Arguments:
        config {Union[str, Dict]} -- Configuration path/dictionary

    Keyword Arguments:
        inheritance_key {str} -- String used for inheritance paths (default: {'FROM'})

    Returns:
        Dict[str, Any] -- Configuration dictionary
    """
    if isinstance(config, str):
        config_dict = yaml.safe_load(open(config))
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError(f'Expected config to be a str or dict but got {type(config)}.')

    if inheritance_key in config_dict:
        for yaml_file in config_dict[inheritance_key]:
            parent_config = load_config(yaml_file, inheritance_key)
            parent_config.update(config_dict)
            config_dict = parent_config

    return config_dict


def get_logger() -> logging.RootLogger:
    """Generates logger

    Returns:
        logging.RootLogger -- Logger
    """
    logger = logging.getLogger()
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s: %(levelname)s - %(message)s")
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def load_checkpoint(model, optimizer, checkpoint_path: str, overwrite=False):
    """Load checkpoint

    Arguments:
        model {[type]} -- PyTorch model  # TODO
        optimizer {[type]} -- Optimizer  # TODO
        checkpoint_path {str} -- Path of checkpoint

    Keyword Arguments:
        overwrite {bool} -- Whether to overwrite local file when using Weights & Biases (default: {False})

    Returns:
        [type] -- [description]  # TODO
    """
    # Load model and optimizer states
    if 'wandb' in checkpoint_path:
        # Get state dict from Weights & Biases
        wandb_dir, checkpoint_name = checkpoint_path.rsplit('/', 1)
        checkpoint_path = wandb.restore(checkpoint_name, run_path=wandb_dir, replace=overwrite).name
        checkpoint = torch.load(checkpoint_path)
    else:
        # Get local state dict
        checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


class SaveCheckpoints:
    """Saves checkpoints of training state.
    """

    def __init__(self, fold_id: int = 0, save_path: str = None, num_checkpoints_to_save: int = 5, logger=None) -> None:
        """Saves multiple model checkpoints.

        Keyword Arguments:
            fold_id {int} -- Cross-validation fold ID
            save_path {str} -- Path to save checkpoints (default: {None})
            num_checkpoints_to_save {int} -- Number of checkpoints to save (default: {5})
            logger {logging.RootLogger} -- Logger (default: {None})
        """
        self.save_path = save_path if save_path else os.getcwd()
        self.fold_id = fold_id
        self.best_metrics = {i: -np.inf for i in range(num_checkpoints_to_save)}
        self.paths = {i: None for i in range(num_checkpoints_to_save)}
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.logger = logger

    def __call__(self, model, performance: float, optimizer, epoch: int, loss: float) -> None:
        """Saves checkpoints if they outperform previous metric score. 
        Assumes improved metrics increase in value. 

        Arguments:
            model {[type]} -- PyTorch model
            performance {float} -- Metric score
            optimizer {[type]} -- Optimizer  # TODO
            epoch {int} -- Training epoch
            loss {float} -- Loss function value
        """
        best_metrics_vals = np.array(list(self.best_metrics.values()))
        cond = performance > best_metrics_vals
        if cond.any():
            idx = np.argmin(best_metrics_vals[cond])
            self.logger.info(
                f'Updating model {idx} with score: {performance:.5f} (Previous: {self.best_metrics[idx]:.5f}).')
            self.best_metrics[idx] = performance

            if self.paths[idx]:
                os.remove(self.paths[idx])

            path = f'{self.save_path}/model_fold_{self.fold_id}_idx_{idx}_perf_{round(performance, 5)}.pth'
            # path = f'{wandb.run.dir}/model_fold_{self.fold_id}_idx_{idx}_perf_{round(performance, 5)}.pth'
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, path)
            self.paths[idx] = path
            self.logger.info(f'Best metric values: {self.best_metrics}')
