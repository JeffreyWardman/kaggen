from typing import Any, Dict

import pytorch_lightning as pl
from torch import nn

from . import typed
from .utils import load_func


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model_args: Dict[str, Any],
        loss_args: Dict[str, Any],
        optimizer_args: Dict[str, Any],
        scheduler_args: Dict[str, Any],
        metrics_args: Dict[str, Any],
    ):
        super().__init__()

        # Set up model
        if "pretrained" in model_args["kwargs"] and model_args["kwargs"]["pretrained"]:
            num_classes = model_args["kwargs"]["num_classes"]
            del model_args["kwargs"]["num_classes"]
            self.model = typed.ImportFuncConfig(**model_args).function
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            self.model = typed.ImportFuncConfig(**model_args).function

        self.loss = typed.ImportFuncConfig(**loss_args).function
        self.optimizer = typed.OptimizerConfig(model_params=self.model.parameters(), **optimizer_args).function
        self.scheduler = typed.SchedulerConfig(optimizer=self.optimizer, **scheduler_args).function
        self.scheduler = {"scheduler": self.scheduler, "interval": "epoch", "monitor": "train_loss"}

        self.metrics = self.get_metrics_dict(metrics_args)

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return y, y_hat, loss

    def training_step(self, batch, batch_idx):
        *_, loss = self.shared_step(batch)  # TODO add mixup via kornia/tormentor
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y, y_hat, loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_metrics(self.metrics, y=y, y_hat=y_hat)
        return loss

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def get_metrics_dict(self, metrics_args: Dict[str, Any]):
        metrics_dict = {}
        for metric, args in metrics_args.items():
            metrics_dict[metric] = load_func(**args, call_function=False)
        return metrics_dict

    def log_metrics(self, metrics_dict: Dict[str, Any], y, y_hat, step: str = "val"):
        for metric_name, metric_func in metrics_dict.items():
            if "accuracy" in metric_name:
                y_target = y.argmax(axis=1)
                y_pred = y_hat.argmax(axis=1)
            else:
                y_target = y
                y_pred = y_hat
            self.log(
                f"{step}_{metric_name}",
                metric_func(y_target.cpu(), y_pred.cpu()),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def on_epoch_end(self):
        print("\n")
