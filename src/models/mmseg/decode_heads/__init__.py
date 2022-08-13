from .uper_head import UPerHead
from .segformer_head import SegformerHead
from .fpn_head import FPNHead

HEADS = {
    "UPerHead": UPerHead,
    "SegformerHead": SegformerHead,
    "FPNHead": FPNHead
}
