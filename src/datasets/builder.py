
from copy import deepcopy
from omegaconf import OmegaConf
import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_trans(trans, T = A):
    if trans is None: return None
    trans = deepcopy(trans)
    if OmegaConf.is_list(trans):
        return T.Compose([build_trans(_, T = T) for _ in trans])
    elif trans["type"] in ("OneOf", "SomeOf"):
        return getattr(T, trans.pop("type"))([build_trans(_, T = T) for _ in trans.pop("transforms")], **trans)
    elif trans["type"] == "ToTensorV2":
        trans.pop("type")
        return ToTensorV2(**trans)
    else:
        return getattr(T, trans.pop("type"))(**trans)