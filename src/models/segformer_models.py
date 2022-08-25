import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

class Segformer(nn.Module):
    def __init__(self,  
                model_name = "b0",
                pretrained = False,
                config_path = None,
                num_classes = 1,
                **kwargs
                ):
        super().__init__()
        if pretrained:
            self.model = transformers.SegformerForSemanticSegmentation.from_pretrained(f"nvidia/segformer-{model_name}-finetuned-ade-512-512")
        else:
            self.model = transformers.SegformerForSemanticSegmentation(
                transformers.SegformerConfig.from_pretrained(config_path
            ))
        self.model.decode_head.classifier = nn.Conv2d(
            self.model.decode_head.classifier.in_channels, 
            num_classes, kernel_size=(1, 1), stride=(1, 1))
        
    def forward(self, x):
        out = self.model(x).logits
        out = F.interpolate(
                    out, 
                    size = x.shape[-2:], 
                    mode = "bilinear", 
                    align_corners = False
                )
        return out