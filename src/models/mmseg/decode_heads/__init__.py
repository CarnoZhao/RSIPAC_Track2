from .uper_head import UPerHead
from .segformer_head import SegformerHead

HEADS = {
    "UPerHead": UPerHead,
    "SegformerHead": SegformerHead
}
