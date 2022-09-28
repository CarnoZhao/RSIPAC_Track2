from .base_dataset import BaseData
from .concat_dataset import ConcatData
from .siamese_dataset import SiameseData

datasets = {_.__name__: _ for _ in [
    BaseData,
    ConcatData,
    SiameseData
]}
