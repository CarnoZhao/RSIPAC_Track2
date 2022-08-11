import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from .losses import losses as registry

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

def _get_loss(cfg):
    cfg = cfg.copy()
    loss_type = cfg.pop("type")
    if loss_type.startswith("nn."):
        return getattr(nn, loss_type[3:])(**cfg)
    else:
        return registry[loss_type](**cfg)

def get_loss(cfg):
    cfg = cfg.copy()
    if not OmegaConf.is_list(cfg):
        cfg = [cfg]
    return MultiLoss(cfg)
