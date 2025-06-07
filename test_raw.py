import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from preprocessing.normalize import get_mnist_transform, get_cifar10_transform
from models.mnist_cnn import MNIST_CNN
from models.cifar10_cnn import CIFAR10_CNN
from models.mnist_res import MNIST_ResNet
from models.cifar10_res import CIFAR10_ResNet
from models.mnist_mlp import MNIST_MLP
from models.cifar10_mlp import CIFAR10_MLP


def main():
    # device selection
    device = torch.device(
        "mps"   if torch.backends.mps.is_available() else
        "cuda"  if torch.cuda.is_available()    else
        "cpu"
    )

    # where to write our raw‐model results
    os.makedirs('results/raw', exist_ok=True)

    # which datasets and model‐types to evaluate
    DATASETS    = ['mnist', 'cifar10']
    MODEL_TYPES = ['baseline', 'res', 'mlp']

    # mapping from model‐type+dataset to class and checkpoint filename
    model_map = {
        'baseline': {
            'mnist':    MNIST_CNN,
            'cifar10':  CIFAR10_CNN
        },
        'res': {
            'mnist':    MNIST_ResNet,
            'cifar10':  CIFAR10_ResNet
        },
        'mlp': {
            'mnist':    MNIST_MLP,
            'cifar10':  CIFAR10_MLP
        }
    }
    ckpt_fmt = {
        'baseline': "{ds}_baseline.pt",
        'res':      "{ds}_res.pt",
        'mlp':      "{ds}_mlp.pt"
    }

    # transforms for each dataset
    transforms_map = {
        'mnist':   get_mnist_transform(),
        'cifar10': get_cifar10_transform()  # includes Resize(64×64) + ToTensor
    }

    BATCH_SIZE = 64

    for ds in DATASETS:
        # load test‐set once per dataset
        transform = transforms_map[ds]
        if ds == 'mnist':
            test_set = datasets.MNIST('data', train=False, download=False, transform=transform)
        else:
            test_set = datasets.CIFAR10('data', train=False, download=False, transform=transform)

        test_loader = DataLoader(
            test_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        for mtype in MODEL_TYPES:
            cls = model_map[mtype][ds]
            ckpt = ckpt_fmt[mtype].format(ds=ds)

            if not os.path.exists(ckpt):
                print(f"[skipping] checkpoint not found: {ckpt}")
                continue

            # load model
            model = cls().to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()

            # evaluate
            y_true, y_pred = [], []
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

            # report
            msg = (
                f"{ds} {mtype} | "
                f"Acc={acc:.4f}, F1={f1:.4f}, Rec={rec:.4f}, Prec={prec:.4f}"
            )
            print(msg)

            # write to results/raw/<dataset>_<modeltype>_results.txt
            out_path = os.path.join('results/raw', f"{ds}_{mtype}_results.txt")
            with open(out_path, 'a') as f:
                f.write(msg + "\n")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
