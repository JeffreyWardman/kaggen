from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import UCF101

from .datasets import AlbumentationCIFAR10Dataset as CIFAR10


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        image_shape: Tuple[int, int, int],
        train_transforms=None,
        val_transforms=None,
        label_smoothing: float = 0.2,
        batch_size: int = 64,
        num_workers: int = 8,
        use_collate_fn: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.dims = image_shape
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.num_workers = num_workers
        self.use_collate_fn = use_collate_fn
        self.num_classes = 10

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(
                root=self.data_dir, train=True, augmentations=self.val_transforms, label_smoothing=self.label_smoothing
            )
            self.train_dataset, self.val_dataset = random_split(cifar_full, [45000, 5000])
            self.train_dataset.augmentations = self.train_transforms

        if stage == "test" or stage is None:
            self.test_dataset = CIFAR10(root=self.data_dir, train=False, augmentations=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn if self.use_collate_fn else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn if self.use_collate_fn else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn if self.use_collate_fn else None,
        )

    def collate_fn(self, batch):
        images = []
        targets = []
        for image, target in batch:
            images.append(np.array(image))
            targets.append(target)
        return torch.Tensor(images).permute(0, 3, 1, 2), torch.Tensor(targets).type(torch.int64)


class UCF101DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        annotation_path: str,
        image_shape: Tuple[int, int, int],
        transform_cfg=None,
        batch_size: int = 64,
        num_workers: int = 8,
        use_collate_fn: bool = False,
    ):
        super(UCF101DataModule, self).__init__()
        self.data_dir = data_dir
        self.annotation_path = annotation_path
        self.image_shape = image_shape
        self.train_transforms = transform_cfg  # TODO get transforms
        self.val_transforms = transform_cfg  # TODO get transforms
        self.dims = image_shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_collate_fn = use_collate_fn

    def prepare_data(self):
        pass

    def setup(self, stage: None):
        if stage == "fit" or stage is None:
            self.trainval_dataset = UCF101(
                root=self.data_dir,
                annotation_path=self.annotation_path,
                frames_per_clip=32,
                num_workers=self.num_workers,
                transform=self.train_transforms,
                train=True,
            )
            self.train_dataset, self.val_dataset = random_split(
                self.trainval_dataset, lengths=[0.8 * len(self.trainval_dataset), 0.2 * len(self.trainval_dataset)]
            )
            self.val_dataset.augmentations = self.val_transforms

        if stage == "test" or stage is None:
            # Same as val_dataset
            self.test_dataset = UCF101(
                root=self.data_dir,
                annotation_path=self.annotation_path,
                frames_per_clip=32,
                num_workers=self.num_workers,
                transform=self.val_transforms,
                train=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=True,
            collate_fn=self.collate_fn if self.use_collate_fn else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=True,
            collate_fn=self.collate_fn if self.use_collate_fn else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=True,
            collate_fn=self.collate_fn if self.use_collate_fn else None,
        )

    def collate_fn(self, batch):
        images1 = []
        images2 = []
        targets = []
        for i, ((image1, image2), _, target) in enumerate(batch):
            images1.append(image1)
            images2.append(image2)
            targets.append(target)

        # Transform already reshaped image to BxCxTxHxW
        images1 = torch.stack(images1).type(torch.float)
        images2 = torch.stack(images2).type(torch.float)
        return (images1, images2, torch.Tensor([])), torch.Tensor(targets).type(torch.int64)
