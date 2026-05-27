import torch
import torch.nn as nn

from models.multi_faces_resnet import BasicBlock, FaceAttributeDecoder, ResNet
from models.multi_lenet import MultiLeNetO, MultiLeNetR
from models.pspnet import SegmentationDecoder, get_segmentation_encoder


def get_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_parallel = cfg["parallel"] and torch.cuda.is_available()
    data = cfg["dataset"]

    if "mnist" in data:
        model = {}
        model["rep"] = MultiLeNetR()
        if use_parallel:
            model["rep"] = nn.DataParallel(model["rep"])
        model["rep"].to(device)
        if "L" in cfg["tasks"]:
            model["L"] = MultiLeNetO()
            if use_parallel:
                model["L"] = nn.DataParallel(model["L"])
            model["L"].to(device)
        if "R" in cfg["tasks"]:
            model["R"] = MultiLeNetO()
            if use_parallel:
                model["R"] = nn.DataParallel(model["R"])
            model["R"].to(device)
        return model, device

    if "cityscapes" in data:
        model = {}
        model["rep"] = get_segmentation_encoder()
        if use_parallel:
            model["rep"] = nn.DataParallel(model["rep"])
        model["rep"].to(device)
        if "S" in cfg["tasks"]:
            model["S"] = SegmentationDecoder(
                num_class=cfg.get("num_classes", 19), task_type="C"
            )
            if use_parallel:
                model["S"] = nn.DataParallel(model["S"])
            model["S"].to(device)
        if "I" in cfg["tasks"]:
            model["I"] = SegmentationDecoder(num_class=2, task_type="R")
            if use_parallel:
                model["I"] = nn.DataParallel(model["I"])
            model["I"].to(device)
        if "D" in cfg["tasks"]:
            model["D"] = SegmentationDecoder(num_class=1, task_type="R")
            if use_parallel:
                model["D"] = nn.DataParallel(model["D"])
            model["D"].to(device)
        return model, device

    if "celeba" in data:
        model = {}
        model["rep"] = ResNet(BasicBlock, [2, 2, 2, 2])
        if use_parallel:
            model["rep"] = nn.DataParallel(model["rep"])
        model["rep"].to(device)
        for t in cfg["tasks"]:
            model[t] = FaceAttributeDecoder(input_dim=512)
            if use_parallel:
                model[t] = nn.DataParallel(model[t])
            model[t].to(device)
        return model, device
