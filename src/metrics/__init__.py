from .dice_metric import DiceMetric
from .classification_metric import ClassificationMetric

metrics = {_.__name__: _ for _ in [
    DiceMetric, ClassificationMetric,
]}