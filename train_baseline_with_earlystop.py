# src/training/train_baseline.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt

from preprocessing.normalize import get_mnist_transform, get_cifar10_transform
from models.mnist_cnn import MNIST_CNN
from models.cifar10_cnn import CIFAR10_CNN
from training.utils import train_one_epoch, evaluate
#----------------------------------------------
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context
#----------------------------------------------

# configs
DATASET = "mnist"    # choose "mnist" or "cifar10"
BATCH_SIZE = 64       # mini-batch size
EPOCHS = 300           # number of training epochs
LR = 0.0003            # learning rate for Adam optimizer

# cpu/gpu setup
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #nvidia
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # macOS

# data directory
os.makedirs("data", exist_ok=True)

# load dataset and transform (EDA)
if DATASET == "mnist":
    transform = get_mnist_transform()
    train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("data", train=False, download=True, transform=transform)
    model = MNIST_CNN().to(device)
else:
    transform = get_cifar10_transform()
    train_set = datasets.CIFAR10("data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10("data", train=False, download=True, transform=transform)
    model = CIFAR10_CNN().to(device)

# dataLoaders
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Lists for tracking metrics
train_losses = []
test_accuracies = []
#----------------------------------------------
PATIENCE = EPOCHS*0.1             # epochs to wait for improvement before stopping
best_val_loss = float('inf')
epochs_no_improve = 0
#----------------------------------------------
# training loop
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, device, test_loader, criterion)
    train_losses.append(train_loss)
    test_accuracies.append(test_acc)
    print(f"Epoch {epoch}/{EPOCHS}  "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # ---------------------------------------
    # --- early stopping check ---
    if test_loss < best_val_loss:
        best_val_loss = test_loss
        epochs_no_improve = 0
        # save checkpoint of best model
        torch.save(model.state_dict(), f"{DATASET}_cnn_best.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs "
                  f"(no improvement in last {PATIENCE} epochs).")
            break
    #---------------------------------------
#----------------------------------------
epochs_ran = len(train_losses)
#----------------------------------------
# Plot training loss over epochs
plt.figure()
plt.plot(range(1, epochs_ran + 1), train_losses)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot test accuracy over epochs
plt.figure()
plt.plot(range(1, epochs_ran + 1), test_accuracies)
plt.title('Test Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Save model
torch.save(model.state_dict(), f"{DATASET}_cnn.pt")
