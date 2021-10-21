from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch.nn as nn

class DeepLabV3Plus_ResNet101(nn.Module):
    def __init__(self, num_classes=11):
        super(DeepLabV3Plus_ResNet101, self).__init__()

        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                        progress=True)
        self.model.classifier = DeepLabHead(2048, num_classes)
    
    def forward(self, x):
        print(self.model)
        return self.model(x)['out']