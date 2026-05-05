import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss


class MTLTrainer(nn.Module):
    """Wraps any MTL backbone with per-task cross-entropy loss computation."""

    def __init__(self, model, init_weight):
        super(MTLTrainer, self).__init__()
        self.model = model
        self.weights = nn.Parameter(torch.from_numpy(init_weight).float())
        self.ce_loss = CrossEntropyLoss()

    def forward(self, x, ts):
        n_tasks = 2
        ys = self.model(x)
        task_loss = []
        for i in range(n_tasks):
            task_loss.append(self.ce_loss(ys[:, i], ts[:, i]))
        return torch.stack(task_loss)
