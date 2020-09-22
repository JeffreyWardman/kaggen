import importlib

import efficientnet_pytorch
import torch
import torch.nn.functional as F
import torchvision.models
from torch import nn


class PremadeModel(torch.nn.Module):

    def __init__(self, model_name: str, module: str, num_classes: int, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.module_name = module
        self.module = importlib.import_module(module)
        self.pretrained = pretrained

        self.backend = self.get_backend()
        self.fc = torch.nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backend(x)
        x = self.fc(x)
        return x

    def get_backend(self):
        if 'efficientnet' in self.model_name.lower():
            return efficientnet_pytorch.EfficientNet.from_pretrained(self.model_name)
        elif 'resnet' in self.model_name.lower():
            return getattr(self.module, self.model_name)(pretrained=self.pretrained)
        else:
            raise ValueError(f'{self.model_name} not found in {self.module}')


class BasicModel(nn.Module):

    def __init__(self, model_name: str, module: str, num_classes: int, pretrained=True):
        super(BasicModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
