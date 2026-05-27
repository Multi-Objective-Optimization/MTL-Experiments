from functools import partial

import torch.nn.functional as F


def nll(pred, gt, val=False):
    if val:
        return F.nll_loss(pred, gt, reduction="sum")
    else:
        return F.nll_loss(pred, gt)


def cross_entropy2d(input, target, weight=None, val=False, ignore_index=250):
    if val:
        size_average = False
    else:
        size_average = True

    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    valid_mask = (target >= 0) & (target != ignore_index)
    log_p = log_p[valid_mask.view(n * h * w, 1).repeat(1, c)]
    log_p = log_p.view(-1, c)

    target = target[valid_mask]
    if target.numel() < 1:
        return input.sum() * 0.0
    loss = F.nll_loss(
        log_p, target, ignore_index=ignore_index, weight=weight, reduction="sum"
    )
    if size_average:
        loss /= valid_mask.sum()
    return loss


def l1_loss_depth(input, target, val=False):
    if val:
        size_average = False
    else:
        size_average = True
    mask = target != 0
    if mask.sum() < 1:
        return input.sum() * 0.0

    lss = F.l1_loss(input[mask], target[mask], reduction="sum")
    if size_average:
        lss = lss / mask.sum()
    return lss


def l1_loss_instance(input, target, val=False, ignore_index=250):
    if val:
        size_average = False
    else:
        size_average = True
    mask = target != ignore_index
    if mask.sum() < 1:
        return input.sum() * 0.0

    lss = F.l1_loss(input[mask], target[mask], reduction="sum")
    if size_average:
        lss = lss / mask.sum()
    return lss


def get_loss(cfg):
    if "mnist" in cfg["dataset"]:
        return {t: nll for t in cfg["tasks"]}

    if "cityscapes" in cfg["dataset"]:
        ignore_index = cfg.get("ignore_index", 250)
        loss_fn = {}
        if "S" in cfg["tasks"]:
            loss_fn["S"] = partial(cross_entropy2d, ignore_index=ignore_index)
        if "I" in cfg["tasks"]:
            loss_fn["I"] = partial(l1_loss_instance, ignore_index=ignore_index)
        if "D" in cfg["tasks"]:
            loss_fn["D"] = l1_loss_depth
        return loss_fn

    if "celeba" in cfg["dataset"]:
        return {t: nll for t in cfg["tasks"]}
