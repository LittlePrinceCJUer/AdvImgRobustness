import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from preprocessing.normalize import get_mnist_transform, get_cifar10_transform
from models.mnist_cnn import MNIST_CNN
from models.cifar10_cnn import CIFAR10_CNN
from training.utils import train_one_epoch, evaluate
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.deepfool import deepfool_attack
from attacks.cw import cw_l2_attack

# Configuration
DATASET = 'mnist'  # or 'cifar10'
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
ATTACK = 'fgsm'
EPS_PARAMS = {
    'fgsm': {'epsilon':0.3},
    'pgd': {'epsilon':0.03,'alpha':0.01,'iters':40},
    'deepfool': {},
    'cw': {'c':1e-2,'kappa':0,'max_iter':100,'lr':0.01}
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
attack_fn = {'fgsm': fgsm_attack, 'pgd': pgd_attack, 'deepfool': deepfool_attack, 'cw': cw_l2_attack}[ATTACK]

os.makedirs('data', exist_ok=True)
if DATASET == 'mnist':
    transform = get_mnist_transform()
    train_set = datasets.MNIST('data', train=True, download=False, transform=transform)
    test_set = datasets.MNIST('data', train=False, download=False, transform=transform)
    model = MNIST_CNN().to(device)
else:
    transform = get_cifar10_transform()
    train_set = datasets.CIFAR10('data', train=True, download=False, transform=transform)
    test_set = datasets.CIFAR10('data', train=False, download=False, transform=transform)
    model = CIFAR10_CNN().to(device)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def adversarial_training():
    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            adv_data = attack_fn(model, data, target, **EPS_PARAMS[ATTACK])
            inputs = torch.cat([data, adv_data], dim=0)
            labels = torch.cat([target, target], dim=0)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, device, DataLoader(test_set, batch_size=BATCH_SIZE), criterion)
        print(f"Epoch {epoch}/{EPOCHS}  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    # final evaluation
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for data, target in DataLoader(test_set, batch_size=BATCH_SIZE):
            data, target = data.to(device), target.to(device)
            preds = model(data).argmax(dim=1)
            y_true += target.cpu().tolist()
            y_pred += preds.cpu().tolist()
    print(f"Post-adversarial training ({ATTACK} on {DATASET}): Acc={accuracy_score(y_true,y_pred):.4f}, F1={f1_score(y_true,y_pred,average='macro'):.4f}, Rec={recall_score(y_true,y_pred,average='macro'):.4f}, Prec={precision_score(y_true,y_pred,average='macro'):.4f}")
    torch.save(model.state_dict(), f"{DATASET}_{ATTACK}_adv_trained.pt")

if __name__ == '__main__':
    adversarial_training()
