from .smp_models import SMPModel
from .segformer_models import Segformer
from .mmseg_models import MMSegModel
from .monai_models import MonaiModel

models = {
    "SMPModel": SMPModel,
    "Segformer": Segformer,
    "MMSegModel": MMSegModel,
    "MonaiModel": MonaiModel
}