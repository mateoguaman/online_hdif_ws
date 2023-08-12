import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

    def forward(self, x):
        out = self.model(x)
        return out