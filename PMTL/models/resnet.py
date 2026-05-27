import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, n_tasks):
        super(ResNet18, self).__init__()
        self.n_tasks = n_tasks
        self.feature_extractor = models.resnet18(weights=None)
        self.feature_extractor.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        fc_in_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fc_in_features, 100)
        for i in range(self.n_tasks):
            setattr(self, "task_{}".format(i), nn.Linear(100, 10))

    def forward(self, x):
        x = F.relu(self.feature_extractor(x))
        outs = []
        for i in range(self.n_tasks):
            layer = getattr(self, "task_{}".format(i))
            outs.append(layer(x))
        return torch.stack(outs, dim=1)
