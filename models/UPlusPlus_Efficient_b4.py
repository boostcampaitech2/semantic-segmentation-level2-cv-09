import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UPlusPlus_Efficient_b4(nn.Module):
    def __init__(self, num_classes=11):
        super(UPlusPlus_Efficient_b4, self).__init__()
        
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b5",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation = "softmax2d"
        )
    
    def forward(self, x):
        return self.model(x)
    