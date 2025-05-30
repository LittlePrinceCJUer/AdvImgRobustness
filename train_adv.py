import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from preprocessing.normalize import get_mnist_transform
from models.mnist_cnn import MNIST_CNN
from models.cifar10_cnn import CIFAR10_CNN
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.deepfool import deepfool_attack
from attacks.cw import cw_l2_attack

def main():
    # device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # configs
    DATASETS = ['mnist', 'cifar10']
    ATTACK_FUNCS = {
        'fgsm':     fgsm_attack,
        'pgd':      pgd_attack,
        'deepfool': deepfool_attack,
        'cw':       cw_l2_attack
    }
    ATTACK_PARAMS = {
        'fgsm':     {'epsilon': 0.3},
        'pgd':      {'epsilon': 0.03, 'alpha': 0.005, 'iters': 20},
        'deepfool': {'overshoot': 0.02, 'max_iter': 10},
        'cw':       {'c': 0.01, 'kappa': 0, 'max_iter': 50, 'lr': 0.01}
    }
    BATCH_SIZE = 64
    EPOCHS     = 40
    LR         = 1e-3

    # ensure output directories exist
    os.makedirs('advResults/logs/baseline',  exist_ok=True)
    os.makedirs('advResults/models/baseline', exist_ok=True)

    # train loop
    for DATASET in DATASETS:
        for ATTACK, attack_fn in ATTACK_FUNCS.items():
            EPS_PARAMS = ATTACK_PARAMS[ATTACK]

            # log
            log_path = os.path.join(
                'advResults', 'logs', 'baseline',
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
                model_cls = MNIST_CNN
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
                model_cls = CIFAR10_CNN

            train_loader = DataLoader(
                train_set, batch_size=BATCH_SIZE,
                shuffle=True,  num_workers=4
            )
            test_loader  = DataLoader(
                test_set,  batch_size=BATCH_SIZE,
                shuffle=False, num_workers=4
            )

            model     = model_cls().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            # main training loop
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
            prec = precision_score(
                y_true, y_pred, average='macro', zero_division=0
            )

            msg_orig = (
                f"Post-adversarial training on original test set "
                f"({ATTACK} on {DATASET}): "
                f"Acc={acc:.4f}, F1={f1:.4f}, Rec={rec:.4f}, Prec={prec:.4f}"
            )
            print(msg_orig)
            with open(log_path, 'a') as f:
                f.write(msg_orig + "\n")

            # Final evaluation: extended test set
            orig_imgs, orig_labels = [], []
            for data, target in test_loader:
                orig_imgs.append(data)
                orig_labels.append(target)
            orig_imgs   = torch.cat(orig_imgs,   dim=0)
            orig_labels = torch.cat(orig_labels, dim=0)

            adv_dir = os.path.join('data', 'adv', DATASET, ATTACK)
            batched = sorted(
                os.listdir(adv_dir),
                key=lambda fn: int(fn.split('_')[1].split('.')[0])
            )
            adv_batches = [
                torch.load(os.path.join(adv_dir, fn))
                for fn in batched
            ]
            adv_imgs   = torch.cat(adv_batches, dim=0)
            adv_labels = orig_labels.clone()

            ext_imgs   = torch.cat([orig_imgs, adv_imgs],   dim=0)
            ext_labels = torch.cat([orig_labels, adv_labels], dim=0)
            ext_loader = DataLoader(
                TensorDataset(ext_imgs, ext_labels),
                batch_size=BATCH_SIZE, shuffle=False
            )

            y_true_ext, y_pred_ext = [], []
            with torch.no_grad():
                for data, target in ext_loader:
                    data, target = data.to(device), target.to(device)
                    preds = model(data).argmax(dim=1)
                    y_true_ext += target.cpu().tolist()
                    y_pred_ext += preds.cpu().tolist()

            acc_ext  = accuracy_score(y_true_ext, y_pred_ext)
            f1_ext   = f1_score(y_true_ext, y_pred_ext, average='macro')
            rec_ext  = recall_score(y_true_ext, y_pred_ext, average='macro')
            prec_ext = precision_score(
                y_true_ext, y_pred_ext, average='macro', zero_division=0
            )

            msg_ext = (
                f"Post-adversarial training on extended test set "
                f"({ATTACK} on {DATASET}): "
                f"Acc={acc_ext:.4f}, F1={f1_ext:.4f}, Rec={rec_ext:.4f}, Prec={prec_ext:.4f}"
            )
            print(msg_ext)
            with open(log_path, 'a') as f:
                f.write(msg_ext + "\n")

            # Save final model
            model_path = os.path.join(
                'advResults', 'models', 'baseline',
                f"{DATASET}_{ATTACK}_adv_trained.pt"
            )
            torch.save(model.state_dict(), model_path)
            print(f"Completed adversarial training on {DATASET} with {ATTACK}\n")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()