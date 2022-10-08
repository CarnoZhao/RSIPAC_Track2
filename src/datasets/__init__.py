from .seg_dataset import SegData

datasets = {_.__name__: _ for _ in [
    SegData,
]}
