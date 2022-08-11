import torch
import torch.nn as nn
import torch.nn.functional as F

def raise_(ex):
    raise ex

try:
    from monai.networks import nets

    class MonaiModel(nn.Module):
        def __init__(self, model_type, **kwargs):
            super().__init__()
            self.model = getattr(nets, model_type)(**kwargs)

        def forward(self, x):
            return self.model(x)
except:
    MonaiModel = lambda *args, **kwargs: raise_( NotImplementedError())
        