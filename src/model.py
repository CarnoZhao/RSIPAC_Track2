import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import models as registry

def get_model(cfg):
    cfg = cfg.copy()
    model = registry[cfg.pop("type")](**cfg)
    
    if "load_from" in cfg and cfg.load_from is not None:
        stt = torch.load(cfg.load_from, map_location = "cpu")
        stt = stt["state_dict"] if "state_dict" in stt else stt
        model.load_state_dict(stt, strict = False)

    return model

