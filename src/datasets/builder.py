
from copy import deepcopy
from omegaconf import OmegaConf
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .augments import AUGS
AUGS = {_.__name__: _ for _ in AUGS}

def build_trans(trans, T = A):
    if isinstance(T, str):
        T = eval(T)
    if trans is None: return None
    trans = deepcopy(trans)
    if OmegaConf.is_list(trans):
        return T.Compose([build_trans(_, T = T) for _ in trans])
    elif trans["type"] in ("Compose", "OneOf", "SomeOf"):
        return getattr(T, trans.pop("type"))([build_trans(_, T = T) for _ in trans.pop("transforms")], **trans)
    elif trans["type"] == "ToTensorV2":
        trans.pop("type")
        return ToTensorV2(**trans)
    elif trans["type"] in AUGS:
        return AUGS[trans.pop("type")](**trans)
    else:
        return getattr(T, trans.pop("type"))(**trans)