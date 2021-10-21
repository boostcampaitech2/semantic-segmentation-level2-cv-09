import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class Unet_Efficient_b5(nn.Module):
    def __init__(self, num_classes=11):
        super(Unet_Efficient_b5, self).__init__()
        
        self.model = smp.Unet(
            encoder_name="efficientnet-b5",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation = "softmax"
        )
    
    def forward(self, x):
        return self.model(x)
    