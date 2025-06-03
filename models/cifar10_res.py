import torch
import torch.nn as nn
import torchvision.models as models

class CIFAR10_ResNet(nn.Module):
    """
    ResNet-18 backbone adapted for CIFAR-10 (3x64x64 inputs).
    Final FC layer output = 10 classes.
    """
    def __init__(self):
        super(CIFAR10_ResNet, self).__init__()
        # Load a ResNet-18 without pretrained weights
        self.resnet = models.resnet18(pretrained=False)
        # Modify first conv to accept 3×64×64 (ResNet default is 3×224×224; we can keep it but accept 64×64)
        # Optionally, we could reduce kernel_size/stride—but 64×64 still works with default.
        # Change final fully-connected to 10 outputs
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 10)

    def forward(self, x):
        return self.resnet(x)
