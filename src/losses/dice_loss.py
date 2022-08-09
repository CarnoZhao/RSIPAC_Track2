import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, ignore_index = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        if len(target.shape) != len(logits.shape) and logits.shape[1] != 1:
            logits = logits.softmax(1)
            num_classes = logits.shape[1]
            loss = 0
            for c in range(num_classes):
                inte = ((target == c) * logits[:,c]).sum((1,2))

                card = ((target == c).pow(2) + logits[:,c].pow(2)).sum((1,2))

                loss += 2 * inte / (card + 1e-6)
            loss /= num_classes
        else:
            if logits.shape[1] == 1: logits = logits.squeeze(1)
            logits = logits.sigmoid()
            inte = ((target == 1) * logits).sum((1,2))
            card = ((target == 1).pow(2) + logits.pow(2)).sum((1,2))
            loss = 2 * inte / (card + 1e-6)
        return 1 - loss.mean()