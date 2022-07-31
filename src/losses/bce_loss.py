import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEWithIgnoreLoss(nn.Module):
    def __init__(self, ignore_index = 255):
        super().__init__()
        self.ignore_index = ignore_index
        self.bce = nn.BCEWithLogitsLoss(reduction = "none")

    def forward(self, logits, target):
        if len(logits.shape) != len(target.shape) and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        target = target.float()
        ignore_mask = target == self.ignore_index
        target[ignore_mask] = 0

        loss = self.bce(logits, target)
        return loss[~ignore_mask].mean()