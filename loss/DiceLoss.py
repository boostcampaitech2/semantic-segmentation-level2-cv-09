import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.dice = smp.losses.DiceLoss(mode="multiclass", smooth=1)

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        
        # intersection = (inputs * targets).sum()                            
        # dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return self.dice(inputs, targets)