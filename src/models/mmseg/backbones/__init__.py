from .mit import MixVisionTransformer
from .pvt import PVTS

BACKBONES = {
    "MixVisionTransformer": MixVisionTransformer,
    **PVTS
}