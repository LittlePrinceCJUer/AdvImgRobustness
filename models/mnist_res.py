import torch
import torch.nn as nn
import torchvision.models as models

class MNIST_ResNet(nn.Module):
    """
    ResNet-18 backbone adapted for MNIST (1×28×28 inputs). 
    We replace the first convolution to accept 1-channel.
    Final FC layer output = 10 classes.
    """
    def __init__(self):
        super(MNIST_ResNet, self).__init__()
        # Load a ResNet-18 skeleton
        self.resnet = models.resnet18(weights=None)  # No pretrained weights
        # Replace the first conv layer to accept 1 channel instead of 3
        self.resnet.conv1 = nn.Conv2d(
            1,                                     # in_channels = 1 for MNIST
            64,                                    # out_channels stays 64
            kernel_size=7, stride=2, padding=3, bias=False
        )
        # Since MNIST images are 28×28, after conv1 (stride=2) you get 14×14; suffice for resnet.
        # Change final fully-connected to 10 outputs
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 10)

    def forward(self, x):
        return self.resnet(x)
