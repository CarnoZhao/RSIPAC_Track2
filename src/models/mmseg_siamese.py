import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


import timm
from .mmseg.backbones import BACKBONES
from .mmseg.decode_heads import HEADS
from .mmseg.necks import NECKS
from .mmseg.utils.ops import resize
from .mmseg.blocks.layer_norm import LayerNorm
from .mmseg_models import MMSegModel


class MMSegSiamese(MMSegModel):
    def extract_feat_siamese(self, img):
        x0 = self.extract_feat(img[0])
        x1 = self.extract_feat(img[1])
        return [torch.abs(x0[i] - x1[i]) for i in range(len(x0))]

    def forward(self, img):
        x = self.extract_feat_siamese(img)
        out = self.decode_head.forward_test(x)
        out = resize(
            input=out,
            size=img[0].shape[2:],
            mode='bilinear',
            align_corners=self.decode_head.align_corners)
        if self.with_aux_head and self.training:
            aux_out = self.aux_head.forward_test(x)
            aux_out = resize(
                input=aux_out,
                size=img[0].shape[2:],
                mode='bilinear',
                align_corners=self.aux_head.align_corners)
            return out, aux_out
        return out