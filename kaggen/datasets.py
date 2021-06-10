from typing import Tuple

import torch
from torch.utils.data.dataset import Dataset


class ContrastiveDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        aug1, target = self.dataset.__getitem__(idx)
        aug2, _ = self.dataset.__getitem__(idx)
        return (aug1, aug2), target

    def __len__(self):
        return self.dataset.__len__()
