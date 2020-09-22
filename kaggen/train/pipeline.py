import importlib
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn.utils
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from ..dataset import Collator, ImageDataset
from ..models import PremadeModel
from ..utils import (SaveCheckpoints, get_logger, load_checkpoint, load_config, set_seed)
from .augmentations import augment
from .loaders import (csvs_to_dataloaders, get_dataloader, get_loss_function, get_lr_scheduler, get_model,
                      get_optimizer, get_train_val_datasets)


class TrainPipeline:
    """Model training pipeline.
    """

    def __init__(self,
                 train_config_path: str,
                 class_config_path: str,
                 num_checkpoints_to_save: int = 5,
                 logger: logging.RootLogger = None) -> None:
        """Initializes training pipeline with required arguments.

        Arguments:
            train_config_path {str} -- Path of YAML file that contains training pipeline configurations
            class_config_path {str} -- Path of YAML file that contains dictionary with class names and IDs

        Keyword Arguments:
            num_checkpoints_to_save {int} -- Number of checkpoints to save (default: {5})
            logger {logging.RootLogger} -- Logger (default: {None})
        """
        self.config = load_config(train_config_path)
        self.num_checkpoints_to_save = num_checkpoints_to_save
        self.logger = logger
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        set_seed(self.config['seed'])
        self.classes = {class_name: idx for idx, class_name in enumerate(load_config(class_config_path)['classes'])}
        # self.run = wandb.init(project="kaggle_cornell_birdcall_identification", job_type='train', config=self.config)

        # Load and shuffle train/val dataset
        if self.config['train_csv_path']:
            self.df_train = pd.read_csv(self.config['train_csv_path'], low_memory=False)
            self.df_train = sklearn.utils.shuffle(self.df_train)

    def __call__(self, fold_idx: int = 0, num_checkpoints_to_save: int = 5):
        """Runs training pipeline.

        Keyword Arguments:
            fold_idx {int} -- Fold index (default: {0})
            num_checkpoints_to_save {int} -- Number of models to save (default: {5})
        """
        self.transform = augment(self.config['augmentations'])
        self.save_checkpoints = SaveCheckpoints(fold_id=fold_idx,
                                                save_path=self.config['save_path'],
                                                num_checkpoints_to_save=num_checkpoints_to_save,
                                                logger=self.logger)

        # Get model and optimizer
        self.model = get_model(model_config=self.config['model'], num_classes=len(self.classes))
        self.optimizer = get_optimizer(self.config['optimizer']['module'],
                                       self.config['optimizer']['name'],
                                       self.model,
                                       lr=self.config['lr']['initial'] / self.config['lr']['warmup_factor'])
        if self.config['checkpoint_path']:
            self.model, self.optimizer = load_checkpoint(self.model, self.optimizer, self.config['checkpoint_path'])

        # Load model to device
        if self.device.type == 'cuda':
            self.model.cuda()

        # Log metrics on wandb
        # wandb.watch(self.model)

        # Generate scheduler
        self.scheduler = get_lr_scheduler(self.config['scheduler']['module'],
                                          self.config['scheduler']['name'],
                                          self.optimizer,
                                          num_epochs=self.config['num_epochs'],
                                          warmup_epoch=self.config['lr']['warmup_epoch'],
                                          warmup_factor=self.config['lr']['warmup_factor'])

        # Generate loss function
        self.loss_func = get_loss_function(self.config['loss_func']['module'], self.config['loss_func']['function'])

        # Generate data loaders
        train_set = get_dataloader(transform=self.transform, **self.config['train_data'])
        valid_set = get_dataloader(transform=self.transform, **self.config['valid_data'])
        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=self.config['batch_size'],
                                                        shuffle=True,
                                                        num_workers=self.config['num_workers'])
        self.valid_loader = torch.utils.data.DataLoader(valid_set,
                                                        batch_size=self.config['batch_size'],
                                                        shuffle=False,
                                                        num_workers=self.config['num_workers'])

        # Train model
        self.train_loop()

    def train_loop(self):
        """Training pipeline loop.
        """
        for epoch in range(self.config['num_epochs']):
            logging.info(f'{"":-<10}\nEpoch: {epoch}\n{"":-<10}')
            train_loss = self._train_epoch(loader=self.train_loader)
            val_loss, val_metrics = self._val_epoch(loader=self.valid_loader)
            performance_name, performance = self.config["performance_metric_name"], val_metrics[
                self.config["performance_metric_name"]]
            self.logger.info(
                f'Epoch: {epoch + 1}   Loss: {round(float(train_loss[0]), 5)}   ' + \
                    f'Val Loss: {round(val_loss, 5)}   Val {performance_name}: {round(performance, 5)}'
            )
            # wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'val_avg_f1_micro': val_f1_micro})

            # Save weights
            self.save_checkpoints(self.model,
                                  performance=performance,
                                  optimizer=self.optimizer,
                                  epoch=epoch,
                                  loss=val_loss)

    def _train_epoch(self, loader: DataLoader) -> float:
        """Training step per epoch

        Arguments:
            loader {DataLoader} -- Training set dataloader

        Returns:
            float -- Loss value
        """
        self.model.train()

        train_loss = []
        bar = tqdm(loader)
        for (data, target) in bar:
            # TODO: for amp: https://www.kaggle.com/haqishen/1st-place-soluiton-code-small-ver#Train-&-Valid-Function
            self.optimizer.zero_grad()

            data = data.to(self.device, dtype=torch.float)
            target = target.long().to(self.device)

            logit = self.model(data)  #.sigmoid()
            loss = self.loss_func(logit, target)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            loss_np = loss.detach().cpu().numpy()
            train_loss.append(loss_np)
            smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
            bar.set_description(f'Loss: {float(loss_np):.5f}, Smooth Loss: {smooth_loss:.5f}')
        return train_loss

    def _val_epoch(self, loader: DataLoader) -> Tuple[List[float], List[float]]:
        """Validation step per epoch

        Arguments:
            loader {DataLoader} -- Validation set dataloader

        Returns:
            Tuple[List[float], List[float]] -- Loss and metric
        """
        self.model.eval()
        val_loss = []
        logits = []
        preds = []
        targets = []

        with torch.no_grad():
            for (data, target) in tqdm(loader):
                data = data.to(self.device)
                target = target.to(self.device)
                logit = self.model(data).sigmoid()  # TODO output layer activation in config
                loss = self.loss_func(logit, target)
                pred = logit.detach()

                logits.append(logit)
                preds.append(pred)
                targets.append(target)
                val_loss.append(loss.detach().cpu().numpy())

        val_loss = np.mean(val_loss)
        logits = torch.cat(logits).cpu().numpy()
        preds = torch.cat(preds).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()

        # Metrics
        metrics = {}
        for metric in self.config['metrics']:
            metric_dict = self.config['metrics'][metric]
            metric = getattr(importlib.import_module(metric_dict['module']), metric_dict['name'])

            if metric_dict['parameters']:
                metrics[metric_dict['name']] = metric(targets, preds, **metric_dict['parameters'])
            else:
                metrics[metric_dict['name']] = metric(targets, preds)

        self.logger.info(f'Val Metrics: {metrics}')
        return val_loss, metrics


if __name__ == '__main__':
    train_config_path = 'configs/train.yaml'
    class_config_path = 'configs/classes.yaml'
    logging.getLogger().setLevel(logging.INFO)
    train_pipeline = TrainPipeline(train_config_path, class_config_path, logger=get_logger())
    train_pipeline()
    # Get dataset from wandb
    # artifact = run.use_artifact('bird-dataset:latest', type='dataset')
    # artifact_dir = artifact.download()
    # assert len(os.listdir(artifact_dir)) == 1, 'Wrong file version possibly downloaded from wandb'
    # artifact_name = os.listdir(artifact_dir)[0]
    # train_csv_path = os.path.join(artifact_dir, artifact_name)
