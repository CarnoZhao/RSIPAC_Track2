from .dice_loss import DiceLoss
from .bce_loss import BCEWithIgnoreLoss

losses = {_.__name__: _ for _ in [
    DiceLoss,
    BCEWithIgnoreLoss
]}