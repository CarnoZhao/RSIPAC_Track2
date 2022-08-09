import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp

class SMPModel(nn.Module):
    def __init__(self, 
                model_type = "Unet", 
                model_name = "resnet18",
                pretrained_weight = "imagenet",
                encoder_depth = 5,
                num_classes = 1,
                **kwargs
                ):
        super().__init__()
        decoder_channels = (256, 128, 64, 32, 16)[:encoder_depth]
        self.model = getattr(smp, model_type)(
            model_name, 
            encoder_weights = pretrained_weight, 
            classes = num_classes,
            encoder_depth = encoder_depth,
            decoder_channels = decoder_channels)
        
    def forward(self, x):
        return self.model(x)