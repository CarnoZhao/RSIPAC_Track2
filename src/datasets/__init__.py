from .base_dataset import BaseData
from .concat_dataset import ConcatData

datasets = {_.__name__: _ for _ in [
    BaseData,
    ConcatData,
]}
