from .uper_head import UPerHead
from .segformer_head import SegformerHead
from .fpn_head import FPNHead
from .daformer_head import DAFormerHead
from .fcn_head import FCNHead

HEADS = {_.__name__: _ for _ in [
    UPerHead, SegformerHead,
    FPNHead, DAFormerHead,
    FCNHead,
]}
