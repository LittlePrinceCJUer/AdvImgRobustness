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
    # device selection
    device = torch.device(
        "mps"   if torch.backends.mps.is_available() else
        "cuda"  if torch.cuda.is_available()    else
        "cpu"
    )

    # datasets to test
    DATASETS = ['mnist', 'cifar10']

    # mapping of attack name to function
    ATTACK_FUNCS = {
        'fgsm':     fgsm_attack,
        'pgd':      pgd_attack,
        'deepfool': deepfool_attack,
        'cw':       cw_l2_attack
    }

    # test-time attack parameters for each dataset (MLP suite)
    TEST_ATTACK_PARAMS = {
        'mnist': {
            'fgsm':     {'epsilon': 0.2},
            'pgd':      {'epsilon': 0.25, 'alpha': 0.006, 'iters': 10},
            'deepfool': {'overshoot': 1.8, 'max_iter': 10},
            'cw':       {'c': 0.01, 'kappa': 0, 'max_iter': 10, 'lr': 0.09}
        },
        'cifar10': {
            'fgsm':     {'epsilon': 0.001},
            'pgd':      {'epsilon': 0.001, 'alpha': 0.005, 'iters': 10},
            'deepfool': {'overshoot': 0.2, 'max_iter': 10},
            'cw':       {'c': 0.01, 'kappa': 0, 'max_iter': 10, 'lr': 0.00024}
        }
    }

    BATCH_SIZE = 64

    # ensure output directory for logs exists
    os.makedirs('advResults/logs/mlp', exist_ok=True)

    # loop over all datasets and attacks
    for DATASET in DATASETS:
        for ATTACK, attack_fn in ATTACK_FUNCS.items():
            EPS_PARAMS = TEST_ATTACK_PARAMS[DATASET][ATTACK]

            # prepare log file
            log_path = os.path.join(
                'advResults', 'logs', 'mlp',
                f"{DATASET}_{ATTACK}_test_log.txt"
            )
            setting_line = (
                f"Test Setting: Dataset={DATASET}, Attack={ATTACK}, "
                f"Attack Params={EPS_PARAMS}, Batch Size={BATCH_SIZE}, Device={device}"
            )
            print(setting_line)
            with open(log_path, 'a') as f:
                f.write(setting_line + "\n")

            # data loading
            if DATASET == 'mnist':
                transform_test = get_mnist_transform()
                test_set = datasets.MNIST(
                    'data', train=False, download=False,
                    transform=transform_test
                )
                model_cls = MNIST_MLP
            else:
                transform_test = transforms.Compose([
                    transforms.Resize((64,64)),
                    transforms.ToTensor()
                ])
                test_set = datasets.CIFAR10(
                    'data', train=False, download=False,
                    transform=transform_test
                )
                model_cls = CIFAR10_MLP

            test_loader = DataLoader(
                test_set, batch_size=BATCH_SIZE,
                shuffle=False, num_workers=0
            )

            # load adversarially trained model
            model_path = os.path.join(
                'advResults', 'models', 'mlp',
                f"{DATASET}_{ATTACK}_adv_trained.pt"
            )
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}, skipping.")
                continue

            model = model_cls().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))

            # evaluate on original test set
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
                f"Evaluation on original test set "
                f"({ATTACK} on {DATASET}): "
                f"Acc={acc:.4f}, F1={f1:.4f}, Rec={rec:.4f}, Prec={prec:.4f}"
            )
            print(msg_orig)
            with open(log_path, 'a') as f:
                f.write(msg_orig + "\n")

            # create directory for saving test-set adversarial examples
            adv_dir = os.path.join('data', 'adv', DATASET, ATTACK)
            os.makedirs(adv_dir, exist_ok=True)

            # generate adversarial examples on test set
            y_true_adv, y_pred_adv = [], []
            all_adv_imgs = []

            for i, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)

                # create adversarial batch
                if ATTACK in ('fgsm', 'pgd', 'cw'):
                    adv = attack_fn(model, data, target, **EPS_PARAMS)
                else:
                    adv = attack_fn(model, data, **EPS_PARAMS)

                # save batch for future reference
                torch.save(adv.cpu(), os.path.join(adv_dir, f"batch_{i}.pt"))
                all_adv_imgs.append(adv.cpu())

                # get predictions on adversarial data
                with torch.no_grad():
                    preds = model(adv).argmax(dim=1)
                y_true_adv += target.cpu().tolist()
                y_pred_adv += preds.cpu().tolist()

            # metrics on adversarial-only test set
            acc_adv  = accuracy_score(y_true_adv, y_pred_adv)
            f1_adv   = f1_score(y_true_adv, y_pred_adv, average='macro')
            rec_adv  = recall_score(y_true_adv, y_pred_adv, average='macro')
            prec_adv = precision_score(y_true_adv, y_pred_adv, average='macro', zero_division=0)

            msg_adv_only = (
                f"Evaluation on adversarial-only test set "
                f"({ATTACK} on {DATASET}): "
                f"Acc={acc_adv:.4f}, F1={f1_adv:.4f}, Rec={rec_adv:.4f}, Prec={prec_adv:.4f}"
            )
            print(msg_adv_only)
            with open(log_path, 'a') as f:
                f.write(msg_adv_only + "\n")

            # build extended test set (original + adversarial)
            orig_imgs, orig_labels = [], []
            for data, target in test_loader:
                orig_imgs.append(data)
                orig_labels.append(target)
            orig_imgs   = torch.cat(orig_imgs, dim=0)
            orig_labels = torch.cat(orig_labels, dim=0)
            adv_imgs    = torch.cat(all_adv_imgs, dim=0)
            adv_labels  = orig_labels.clone()

            ext_imgs   = torch.cat([orig_imgs, adv_imgs],   dim=0)
            ext_labels = torch.cat([orig_labels, adv_labels], dim=0)
            ext_loader = DataLoader(
                TensorDataset(ext_imgs, ext_labels),
                batch_size=BATCH_SIZE, shuffle=False
            )

            # evaluate on extended test set
            y_true_ext, y_pred_ext = [], []
            model.eval()
            with torch.no_grad():
                for data, target in ext_loader:
                    data, target = data.to(device), target.to(device)
                    preds = model(data).argmax(dim=1)
                    y_true_ext += target.cpu().tolist()
                    y_pred_ext += preds.cpu().tolist()

            acc_ext  = accuracy_score(y_true_ext, y_pred_ext)
            f1_ext   = f1_score(y_true_ext, y_pred_ext, average='macro')
            rec_ext  = recall_score(y_true_ext, y_pred_ext, average='macro')
            prec_ext = precision_score(y_true_ext, y_pred_ext, average='macro', zero_division=0)

            msg_ext = (
                f"Evaluation on extended test set "
                f"({ATTACK} on {DATASET}): "
                f"Acc={acc_ext:.4f}, F1={f1_ext:.4f}, Rec={rec_ext:.4f}, Prec={prec_ext:.4f}"
            )
            print(msg_ext)
            with open(log_path, 'a') as f:
                f.write(msg_ext + "\n")

            # free up memory before next experiment
            del model
            del test_loader
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            torch.mps.empty_cache()   if device.type == 'mps'  else None
            gc.collect()

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
