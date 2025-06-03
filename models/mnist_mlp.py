import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_MLP(nn.Module):
    """
    Simple MLP for MNIST: input 1×28×28 → flatten → two hidden layers → 10 outputs.
    """
    def __init__(self):
        super(MNIST_MLP, self).__init__()
        self.flatten = nn.Flatten()
        # 1×28×28 = 784 features
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
