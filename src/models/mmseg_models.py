import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


import timm
from .mmseg.backbones import BACKBONES
from .mmseg.decode_heads import HEADS
from .mmseg.utils.ops import resize
from .mmseg.blocks.layer_norm import LayerNorm


class MMSegModel(nn.Module):
    def __init__(self, 
                backbone,
                decode_head,
                **kwargs,
                ):
        super().__init__()
        self.prepare_backbone(backbone)
        self.prepare_head(decode_head)

    def prepare_backbone(self, backbone):
        backbone = backbone.copy()
        self.is_timm = False
        if "type" not in backbone or backbone["type"] == "timm":
            self.prepare_backbone(backbone)
        elif backbone["type"].startswith("mmseg."):
            self.prepare_mmseg_backbone(backbone)
        else:
            raise NotImplementedError()

    def prepare_mmseg_backbone(self, backbone):
        backbone = backbone.copy()
        backbone["type"] = backbone["type"][6:]
        self.backbone = BACKBONES[backbone.pop("type")](**OmegaConf.to_container(backbone))
        self.reduction_index = self.backbone.out_indices
        self.out_channels = [self.backbone.out_channels[i] for i in self.reduction_index]

    def prepare_timm_backbone(self, backbone):
        self.is_timm = True
        backbone = backbone.copy()
        reductions = backbone.get("reductions", [4, 8, 16, 32])
        self.backbone = timm.create_model(backbone.model_name, pretrained = True, features_only = True)
        feature_info = self.backbone.feature_info.info
        feature_reductions = [_["reduction"] for _ in feature_info]
        self.reduction_index = [feature_reductions.index(r) for r in reductions]
        self.out_channels = [feature_info[i]['num_chs'] for i in self.reduction_index]

        for i in range(len(self.out_channels)):
            layer = LayerNorm(self.out_channels[i], eps=1e-6, data_format="channels_first")
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def prepare_head(self, decode_head):
        decode_head = decode_head.copy()
        decode_head["in_channels"] = self.out_channels
        decode_head["in_index"] = list(range(len(self.out_channels)))
        self.decode_head = HEADS[decode_head.pop("type")](**OmegaConf.to_container(decode_head))

    def extract_feat(self, img):
        out = self.backbone(img)
        if self.is_timm:
            out = [getattr(self, f'norm{i}')(out[i]) for i in self.reduction_index]
        return out

    def forward(self, img):
        x = self.extract_feat(img)
        out = self.decode_head.forward_test(x)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.decode_head.align_corners)
        return out

    def load_state_dict(self, state_dict, strict):
        if state_dict["decode_head.conv_seg.weight"].shape != self.state_dict()["decode_head.conv_seg.weight"].shape:
            state_dict.pop("decode_head.conv_seg.weight")
            state_dict.pop("decode_head.conv_seg.bias")
        return super().load_state_dict(state_dict, strict)
