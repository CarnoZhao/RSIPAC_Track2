from .mit import MixVisionTransformer
from .pvt import PyramidVisionTransformerV2
from .coat import CoaT

BACKBONES = {_.__name__: _ for _ in [
    MixVisionTransformer,
    PyramidVisionTransformerV2,
    CoaT,
]}