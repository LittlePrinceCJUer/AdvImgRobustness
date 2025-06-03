import os
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from preprocessing.normalize import get_mnist_transform
from models.mnist_mlp import MNIST_MLP
from models.cifar10_mlp import CIFAR10_MLP
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.deepfool import deepfool_attack
from attacks.cw import cw_l2_attack

def main():
    # device
    device = torch.device(
        "mps"   if torch.backends.mps.is_available() else
        "cuda"  if torch.cuda.is_available()    else
        "cpu"
    )

    # configs
    DATASETS = ['cifar10']    # or ['mnist']
    ATTACK_FUNCS = {
        'fgsm':     fgsm_attack,
        'pgd':      pgd_attack,
        'deepfool': deepfool_attack,
        'cw':       cw_l2_attack
    }
    # CIFAR-10 adversarial-training attack parameters
    ATTACK_PARAMS = {
        'fgsm':     {'epsilon': 0.001},
        'pgd':      {'epsilon': 0.001, 'alpha': 0.005, 'iters': 10},
        'deepfool': {'overshoot': 0.2, 'max_iter': 10},
        'cw':       {'c': 0.01, 'kappa': 0, 'max_iter': 10, 'lr': 0.00024}
    }
    # If MNIST, uncomment & adjust:
    # ATTACK_PARAMS = {
    #     'fgsm':     {'epsilon': 0.2},
    #     'pgd':      {'epsilon': 0.25, 'alpha': 0.006, 'iters': 10},
    #     'deepfool': {'overshoot': 1.8, 'max_iter': 10},
    #     'cw':       {'c': 0.01, 'kappa': 0, 'max_iter': 10, 'lr': 0.09}
    # }

    BATCH_SIZE = 64
    EPOCHS     = 100
    LR         = 1e-3

    # ensure output directories exist
    os.makedirs('advResults/logs/mlp',    exist_ok=True)
    os.makedirs('advResults/models/mlp',  exist_ok=True)

    # train loop over dataset × attack
    for DATASET in DATASETS:
        for ATTACK, attack_fn in ATTACK_FUNCS.items():
            EPS_PARAMS = ATTACK_PARAMS[ATTACK]

            # log
            log_path = os.path.join(
                'advResults', 'logs', 'mlp',
                f"{DATASET}_{ATTACK}_train_log.txt"
            )
            setting_line = (
                f"Setting: Dataset={DATASET}, Attack={ATTACK}, "
                f"Attack Params={EPS_PARAMS}, Batch Size={BATCH_SIZE}, "
                f"Epochs={EPOCHS}, LR={LR}, Device={device}"
            )
            print(setting_line)
            with open(log_path, 'a') as f:
                f.write(setting_line + "\n")

            # data loading
            if DATASET == 'mnist':
                transform_train = get_mnist_transform()
                transform_test  = get_mnist_transform()
                train_set = datasets.MNIST(
                    'data', train=True,  download=False,
                    transform=transform_train
                )
                test_set  = datasets.MNIST(
                    'data', train=False, download=False,
                    transform=transform_test
                )
                model_cls = MNIST_MLP
            else:
                transform_train = transforms.Compose([
                    transforms.Resize((64,64)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(10),
                    transforms.RandomCrop(64, padding=4),
                    transforms.ToTensor()
                ])
                transform_test = transforms.Compose([
                    transforms.Resize((64,64)),
                    transforms.ToTensor()
                ])
                train_set = datasets.CIFAR10(
                    'data', train=True,  download=False,
                    transform=transform_train
                )
                test_set  = datasets.CIFAR10(
                    'data', train=False, download=False,
                    transform=transform_test
                )
                model_cls = CIFAR10_MLP

            train_loader = DataLoader(
                train_set, batch_size=BATCH_SIZE,
                shuffle=True,  num_workers=0
            )
            test_loader  = DataLoader(
                test_set,  batch_size=BATCH_SIZE,
                shuffle=False, num_workers=0
            )

            model     = model_cls().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            # main training loop for this (DATASET, ATTACK)
            for epoch in range(1, EPOCHS + 1):
                model.train()
                running_loss = 0.0
                correct      = 0
                total        = 0

                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)

                    # generate adversarial batch
                    if ATTACK in ('fgsm', 'pgd', 'cw'):
                        adv = attack_fn(model, data, target, **EPS_PARAMS)
                    else:
                        adv = attack_fn(model, data, **EPS_PARAMS)

                    # combine clean + adv
                    combined_data   = torch.cat([data, adv], dim=0)
                    combined_target = torch.cat([target, target], dim=0)

                    optimizer.zero_grad()
                    outputs = model(combined_data)
                    loss    = F.cross_entropy(outputs, combined_target)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * combined_data.size(0)
                    preds        = outputs.argmax(dim=1)
                    correct     += (preds == combined_target).sum().item()
                    total       += combined_data.size(0)

                train_loss = running_loss / total
                train_acc  = correct / total

                # evaluate on clean test set
                model.eval()
                test_loss    = 0.0
                correct_test = 0
                total_test   = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        outputs = model(data)
                        loss    = F.cross_entropy(outputs, target)
                        test_loss    += loss.item() * data.size(0)
                        preds        = outputs.argmax(dim=1)
                        correct_test += (preds == target).sum().item()
                        total_test   += data.size(0)

                test_loss = test_loss / total_test
                test_acc  = correct_test / total_test

                log_line = (
                    f"Epoch {epoch}/{EPOCHS}  "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}  |  "
                    f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
                )
                print(log_line)
                with open(log_path, 'a') as f:
                    f.write(log_line + "\n")

            # Final evaluation: original test set
            y_true, y_pred = [], []
            model.eval()
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    preds = model(data).argmax(dim=1)
                    y_true += target.cpu().tolist()
                    y_pred += preds.cpu().tolist()

            acc  = accuracy_score(y_true, y_pred)
            f1   = f1_score(y_true, y_pred, average='macro')
            rec  = recall_score(y_true, y_pred, average='macro')
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)

            msg_orig = (
                f"Post-adversarial training on original test set "
                f"({ATTACK} on {DATASET}): "
                f"Acc={acc:.4f}, F1={f1:.4f}, Rec={rec:.4f}, Prec={prec:.4f}"
            )
            print(msg_orig)
            with open(log_path, 'a') as f:
                f.write(msg_orig + "\n")

            # Save final model
            model_path = os.path.join(
                'advResults', 'models', 'mlp',
                f"{DATASET}_{ATTACK}_adv_trained.pt"
            )
            torch.save(model.state_dict(), model_path)
            print(f"Completed adversarial training on {DATASET} with {ATTACK}\n")

            # free up memory before next attack
            del model
            del optimizer
            del train_loader
            del test_loader
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            torch.mps.empty_cache()   if device.type == 'mps'  else None
            gc.collect()

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
