import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt

from preprocessing.normalize import get_mnist_transform, get_cifar10_transform
from models.mnist_res import MNIST_ResNet
from models.cifar10_res import CIFAR10_ResNet
from training.utils import train_one_epoch, evaluate

# configs
DATASET = "cifar10"    # choose "mnist" or "cifar10"
BATCH_SIZE = 64       # mini-batch size
EPOCHS = 300           # number of training epochs
LR = 0.001            # learning rate for Adam optimizer

# cpu/gpu setup
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# data directory
os.makedirs("data", exist_ok=True)

# load dataset and transform (basic normalization / resize if needed)
if DATASET == "mnist":
    transform = get_mnist_transform()
    train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("data", train=False, download=True, transform=transform)
    model = MNIST_ResNet().to(device)
else:
    transform = get_cifar10_transform()  # includes Resize(64×64) + ToTensor
    train_set = datasets.CIFAR10("data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10("data", train=False, download=True, transform=transform)
    model = CIFAR10_ResNet().to(device)

# dataLoaders
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Lists for tracking metrics
train_losses = []
test_accuracies = []

# training loop
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, device, test_loader, criterion)
    train_losses.append(train_loss)
    test_accuracies.append(test_acc)
    print(f"Epoch {epoch}/{EPOCHS}  "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


# make dir for saving figures
os.makedirs("figs", exist_ok=True)

# Plot training loss over epochs
plt.figure()
plt.plot(range(1, EPOCHS + 1), train_losses)
plt.title('ResNet Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(f"figs/{DATASET}_res_train_loss.png")  # Save instead of show
plt.close() # free memory

# Plot test accuracy over epochs
plt.figure()
plt.plot(range(1, EPOCHS + 1), test_accuracies)
plt.title('ResNet Test Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig(f"figs/{DATASET}_res_testAcc.png")  # Save instead of show
plt.close() # free memory

# Save model
torch.save(model.state_dict(), f"{DATASET}_res.pt")
