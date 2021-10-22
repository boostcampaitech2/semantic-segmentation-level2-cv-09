import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class DeepLabV3Plus_Xception(nn.Module):
    def __init__(self, num_classes=11):
        super(DeepLabV3Plus_Xception, self).__init__()

        self.model = smp.DeepLabV3Plus(
            encoder_name="tu-xception41",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    
    def forward(self, x):
        return self.model(x)
    