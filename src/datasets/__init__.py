from .seg_dataset import SegData
from .concat_dataset import ConcatData
from .siamese_dataset import SiameseData

datasets = {_.__name__: _ for _ in [
    SegData,
    ConcatData,
    SiameseData
]}
