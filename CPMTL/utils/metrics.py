from typing import Iterable

import numpy as np
import torch
from termcolor import colored
from torch import Tensor


def topk_accuracies(
        output: Tensor,
        label: Tensor,
        ks: Iterable[int] = (1,),
    ):

    assert output.dim() == 2
    assert label.dim() == 1
    assert output.size(0) == label.size(0)

    maxk = max(ks)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    label = label.unsqueeze(1).expand_as(pred)
    correct = pred.eq(label).float()

    accu_list = []
    for k in ks:
        accu = correct[:, :k].sum(1).mean()
        accu_list.append(accu.item())
    return accu_list


def topk_accuracy(
        output: Tensor,
        label: Tensor,
        k: int = 1,
    ):

    return topk_accuracies(output, label, (k,))[0]


@torch.no_grad()
def evaluate(network, dataloader, device, closures, header=''):
    num_samples = 0
    losses = np.zeros(2)
    top1s = np.zeros(2)
    network.train(False)
    for images, labels in dataloader:
        batch_size = len(images)
        num_samples += batch_size
        images = images.to(device)
        labels = labels.to(device)
        logits = network(images)
        losses_batch = [c(network, logits, labels).item() for c in closures]
        losses += batch_size * np.array(losses_batch)
        top1s[0] += batch_size * topk_accuracy(logits[0], labels[:, 0], k=1)
        top1s[1] += batch_size * topk_accuracy(logits[1], labels[:, 1], k=1)
    losses /= num_samples
    top1s /= num_samples

    loss_msg = '[{}]'.format('/'.join([f'{loss:.6f}' for loss in losses]))
    top1_msg = '[{}]'.format('/'.join([f'{top1 * 100.0:.2f}%' for top1 in top1s]))
    msgs = [
        f'{header}:' if header else '',
        'loss', colored(loss_msg, 'yellow'),
        'top@1', colored(top1_msg, 'yellow')
    ]
    print(' '.join(msgs))
    return losses, top1s
