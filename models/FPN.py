import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class FPN(nn.Module):
    def __init__(self, backbone, num_classes=11):
        super(FPN, self).__init__()
        
        self.model = smp.UnetPlusPlus(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation = "softmax2d"
        )
    
    def forward(self, x):
        return self.model(x)
    