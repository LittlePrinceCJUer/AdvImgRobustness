import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 10, kernel_size=3, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(10, 24, kernel_size=3, stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.dropout1 = nn.Dropout2d(0.2)
        self.fc1 = nn.Linear(2 * 2 * 24, 64)
        #self.dropout2 = nn.Dropout2d(0.2)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)
        #x = self.dropout1(x)
        # Flatten the tensor
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        #x = self.dropout2(x)
        x = self.fc2(x)
        return x