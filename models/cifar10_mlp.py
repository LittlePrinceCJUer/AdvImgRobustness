import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10_MLP(nn.Module):
    """
    Simple MLP for CIFAR-10: input 3×32×32 → flatten → two hidden layers → 10 outputs.
    We first resize 32×32→64×64 and then normalize; here we assume transform already did resize/ToTensor.
    """
    def __init__(self):
        super(CIFAR10_MLP, self).__init__()
        self.flatten = nn.Flatten()
        # After transforms we will have 3×64×64 = 12288 features
        self.fc1 = nn.Linear(3 * 64 * 64, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
