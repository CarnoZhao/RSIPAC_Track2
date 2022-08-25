import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob

from .models import models as registry

def get_model(cfg):
    cfg = cfg.copy()
    model = registry[cfg.pop("type")](**cfg)
    
    if "load_from" in cfg and cfg.load_from is not None:
        if not os.path.exists(cfg.load_from):
            load_from = sorted(glob.glob(cfg.load_from))[-1]
        else:
            load_from = cfg.load_from
        stt = torch.load(load_from, map_location = "cpu")
        stt = stt["state_dict"] if "state_dict" in stt else stt
        model_stt = model.state_dict()
        if all(k.startswith("model.") and k[6:] in model_stt for k in stt.keys()):
            stt = {k[6:]: v for k, v in stt.items()}
        if all(k.startswith("module.") and k[7:] in model_stt for k in stt.keys()):
            stt = {k[7:]: v for k, v in stt.items()}
        model.load_state_dict(stt, strict = False)

    return model

