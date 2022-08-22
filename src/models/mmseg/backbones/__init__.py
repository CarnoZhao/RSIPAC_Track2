from .mit import MixVisionTransformer
from .pvt import PyramidVisionTransformerV2
from .pvt_ms import PyramidVisionTransformerV2MS
from .coat import CoaT

BACKBONES = {_.__name__: _ for _ in [
    MixVisionTransformer,
    PyramidVisionTransformerV2,
    PyramidVisionTransformerV2MS,
    CoaT,
]}