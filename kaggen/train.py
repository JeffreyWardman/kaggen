import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import wandb
import yaml
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from . import lit_callbacks, lit_utils, typed, utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrainPipeline:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.seed_everything()

        # Set up data_module
        self.setup_transforms()
        self.data_module = self.setup_dataloaders()

        # Set up model
        self.lit_model = self.setup_model()
        if self.config["pretrained_path"]:
            self.lit_model = self.load_pretrained_model()

        # Set up loggers and callbacks
        self.loggers = self.setup_loggers()
        self.callbacks = self.setup_callbacks()

        # Set up trainer
        self.trainer = self.setup_trainer()

    def __call__(self):
        return self.train()

    def train(self):
        self.trainer.fit(self.lit_model, self.data_module)
        return self.trainer

    def load_config(self, config_path: str):
        config = OmegaConf.load(config_path)
        additional_configs = [OmegaConf.load(x) for x in config["INHERIT"]]
        config = OmegaConf.merge(config, *additional_configs)
        return config

    def seed_everything(self):
        pl.seed_everything(self.config["seed"])

    def setup_transforms(self):
        if "train" in self.config["transforms"]:
            self.train_transforms = utils.get_transforms(self.config["transforms"]["train"])
        else:
            self.train_transforms = None

        if "val" in self.config["transforms"]:
            self.val_transforms = utils.get_transforms(self.config["transforms"]["val"])
        else:
            self.val_transforms = None

    def setup_dataloaders(self):
        return typed.DataModuleConfig(
            train_transforms=self.train_transforms, val_transforms=self.val_transforms, **self.config["data_module"]
        ).function

    def setup_model(self):
        model_config = {f"{key}_args": val for key, val in self.config["lit_model"].items()}
        model_config = typed.LitModelConfig(**model_config)
        return lit_utils.LitModel(**asdict(model_config))

    def load_pretrained_model(self):
        # TODO in lit model load pretrained weights
        if "wandb" in self.config["pretrained_path"]:
            self.config["pretrained_path"] = wandb.restore(self.config["pretrained_path"]).name
        return self.lit_model.load_from_checkpoint(self.config["pretrained_path"])

    def setup_loggers(self):
        wandb_logger = WandbLogger(
            config=self.config, project=self.config["project"], job_type="train", log_model=True, save_code=True
        )
        # wandb_logger.watch(self.lit_model, log='gradients', log_freq=100)
        return [wandb_logger]

    def setup_callbacks(self):
        # Save multiple models
        checkpoint_callback = ModelCheckpoint(
            monitor=self.config["performance_metric"],
            dirpath=f"{self.config['checkpoint_save_path']}/{self.config['lit_model']['model']['func_name']}",
            filename="{epoch:02d}-{val_loss:.5f}",
            save_top_k=self.config["num_checkpoints"],
            mode="min",
        )

        early_stopping_callback = EarlyStopping(monitor=self.config["performance_metric"], patience=3, verbose=True)

        lr_monitor = LearningRateMonitor(logging_interval="step")

        self.data_module.prepare_data()
        self.data_module.setup()

        val_samples = next(iter(self.data_module.val_dataloader()))
        image_logger = lit_callbacks.ImagePredictionLogger(val_samples, len(val_samples))
        return [early_stopping_callback, checkpoint_callback, lr_monitor, image_logger]

    def setup_trainer(self):
        return pl.Trainer(
            logger=self.loggers,
            callbacks=self.callbacks,
            accumulate_grad_batches=10,
            # plugins = 'ddp_sharded',
            # plugins=DeepSpeedPlugin(allgather_bucket_size=5e8, reduce_bucket_size=5e8),
            **self.config["trainer"],
        )


if __name__ == "__main__":
    import os

    # os.environ["WANDB_MODE"] = "dryrun"
    pipeline = TrainPipeline(config_path="configs/experiment/classification_cifar10.yml")
    trainer = pipeline()
