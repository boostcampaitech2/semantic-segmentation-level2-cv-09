from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class FocalLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.focal_loss = smp.losses.FocalLoss(mode="multiclass")

    def forward(self, input_tensor, target_tensor):
        return self.focal_loss(input_tensor, target_tensor)
