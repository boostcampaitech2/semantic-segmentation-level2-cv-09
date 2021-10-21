import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class DeepLabV3Plus_Xception(nn.Module):
    def __init__(self, num_classes=11):
        super(DeepLabV3Plus_Xception, self).__init__()
        
        self.model = smp.create_model(
            arch="DeepLabV3Plus",
            encoder_name="tu-xception41",
            in_channels=3,
            classes=11,
        )
    
    def forward(self, x):
        return self.model(x)
    