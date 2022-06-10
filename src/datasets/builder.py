
from copy import deepcopy
from omegaconf import OmegaConf
import albumentations as A

def build_trans(trans, T = A):
    if trans is None: return None
    trans = deepcopy(trans)
    if OmegaConf.is_list(trans):
        return T.Compose([build_trans(_, T = T) for _ in trans])
    elif trans["type"] == "OneOf":
        trans.pop("type")
        return T.OneOf([build_trans(_, T = T) for _ in trans.pop("transforms")], **trans)
    else:
        return getattr(T, trans.pop("type"))(**trans)