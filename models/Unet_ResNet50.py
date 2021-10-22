import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class Unet_ResNet50(nn.Module):
    def __init__(self, num_classes=11):
        super(Unet_ResNet50, self).__init__()
        
        self.model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation = "softmax2d"
        )
    
    def forward(self, x):
        return self.model(x)
    