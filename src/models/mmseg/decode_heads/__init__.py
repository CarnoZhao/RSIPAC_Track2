from .uper_head import UPerHead
from .segformer_head import SegformerHead
from .fpn_head import FPNHead
from .daformer_head import DAFormerHead

HEADS = {_.__name__: _ for _ in [
    UPerHead, SegformerHead,
    FPNHead, DAFormerHead,
]}
