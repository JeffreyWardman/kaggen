from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydub
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data.dataset import Dataset


class ImageDataset(Dataset):
    """Dataset for image data.

    Arguments:
        Dataset {torch.utils.data.dataset.Dataset} -- Torch Dataset
    """

    def __init__(self,
                 df: pd.DataFrame,
                 path_column: str,
                 target_column: str,
                 classes: Dict[str, int],
                 image_shape: Tuple[int, int] = (256, 256, 3),
                 transform=None) -> None:
        """Image dataset initialization

        Arguments:
            df {pd.DataFrame} -- Dataframe with paths and target values
            path_column {str} -- Column containing data paths
            target_column {str} -- Column containing targets
            classes {Dict[str, int]} -- Class names and index

        Keyword Arguments:
            image_shape {Tuple[int, int]} -- Image shape (Height, Width, Channels) (default: {(256, 256, 3)})
            transform {[type]} -- Augmentations (default: {None})  # TODO
        """

        self.df = df
        self.path_column = path_column
        self.target_column = target_column
        self.image_shape = image_shape
        self.transform = transform
        self.classes = classes
        self.num_classes = len(classes)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Iterator

        Arguments:
            i {int} -- Index of dataframe row

        Returns:
            Tuple[torch.Tensor, torch.Tensor] -- Input data and label
        """
        row = self.df.iloc[i]
        img = plt.imread(row[self.path_column])[:, :, :-1]

        # Resize image
        img = np.resize(img, self.image_shape)

        if self.transform is not None:
            # Augment image
            img = self.transform(img, row.sampling_rate)['image']

        img = np.transpose(img, (2, 0, 1))  # HxWxD -> DxHxW

        # Get label
        labels = np.zeros(self.num_classes, dtype=np.float32)
        labels[self.classes[row[self.target_column]]] = 1.

        return torch.tensor(img, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
