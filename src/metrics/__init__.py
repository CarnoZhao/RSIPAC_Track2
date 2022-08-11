from .dice_metric import DiceMetric
from .classification_metric import ClassificationMetric

metrics = {
    "DiceMetric": DiceMetric,
    "ClassificationMetric": ClassificationMetric   
}