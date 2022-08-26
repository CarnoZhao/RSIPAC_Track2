import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from .losses import losses as registry
from collections import defaultdict

class MultiLoss(nn.Module):
    def __init__(self, loss_cfg):
        super(MultiLoss, self).__init__()
        self.loss_names = []
        self.loss_weights = []
        for i, loss_arg in enumerate(loss_cfg):
            self.loss_names.append(loss_arg.pop("loss_name", f"loss_{i}"))
            self.loss_weights.append(loss_arg.pop("loss_weight", 1))

        self.losses = [_get_loss(loss_arg) for loss_arg in loss_cfg]

    def forward(self, logits, target):
        losses = {}
        for i in range(len(self.losses)):
            loss = self.losses[i](logits, target)
            losses[self.loss_names[i]] = loss * self.loss_weights[i]
        losses["loss"] = sum(losses.values())
        return losses

class MultiInputLoss(nn.Module):
    def __init__(self, func, weight):
        super(MultiInputLoss, self).__init__()
        self.func = func
        self.weight = weight

    def forward(self, inputs_list, target):
        losses = defaultdict(int)
        if not isinstance(inputs_list, tuple):
            inputs_list = (inputs_list,)
        sum_weights = 0
        for idx, (w, inputs) in enumerate(zip(self.weight, inputs_list)):
            loss = self.func(inputs, target)
            for k, v in loss.items():
                losses[k] += v * w
                losses[k + "_" + str(idx)] += v * w
            sum_weights += w
        for k in losses:
            losses[k] /= sum_weights
        return losses


def _get_loss(cfg):
    cfg = cfg.copy()
    loss_type = cfg.pop("type")
    if loss_type.startswith("nn."):
        func = getattr(nn, loss_type[3:])(**cfg)
    else:
        return registry[loss_type](**cfg)

def get_loss(cfg):
    cfg = cfg.copy()
    if OmegaConf.is_dict(cfg) and "losses" in cfg:
        multi_inputs = cfg.pop("multi_inputs", False)
        input_weights = cfg.pop("input_weights", None)
        cfg = cfg.losses
    elif OmegaConf.is_list(cfg):
        multi_inputs = False
    elif OmegaConf.is_dict(cfg) and "losses" not in cfg:
        multi_inputs = cfg.pop("multi_inputs", False)
        input_weights = cfg.pop("input_weights", None)
        cfg = [cfg]

    func = MultiLoss(cfg)
    if multi_inputs:
        func = MultiInputLoss(func, input_weights)
    return func
