from .dice_metric import DiceMetric
from .classification_metric import ClassificationMetric
from .change_det_ap import ChangeAP

metrics = {_.__name__: _ for _ in [
    DiceMetric, ClassificationMetric,
    ChangeAP,
]}