from torchvision import models
import torch.nn as nn

class FCN_ResNet50(nn.Module):
    def __init__(self, backbone, num_classes=11):
        super(FCN_ResNet50, self).__init__()

        self.model = models.segmentation.fcn_resnet50(pretrained=True)

        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    
    def forward(self, x):
        return self.model(x)['out']
