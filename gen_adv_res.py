import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, utils, transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from preprocessing.normalize import get_mnist_transform, get_cifar10_transform
from models.mnist_res import MNIST_ResNet
from models.cifar10_res import CIFAR10_ResNet
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.deepfool import deepfool_attack
from attacks.cw import cw_l2_attack

# --- Evaluation transforms (fixed) ---
mnist_eval_transform = get_mnist_transform()
cifar_eval_transform = transforms.Compose([
    transforms.Resize((64, 64)),  # match training resize
    transforms.ToTensor()         # min-max to [0,1]
])

# --- Data loaders for testset ---
mnist_loader = DataLoader(
    datasets.MNIST('data', train=False, download=False, transform=mnist_eval_transform),
    batch_size=64, shuffle=False
)
cifar_loader = DataLoader(
    datasets.CIFAR10('data', train=False, download=False, transform=cifar_eval_transform),
    batch_size=64, shuffle=False
)

# --- Evaluate & save (ResNet) ---
def evaluate_and_save_res(dataset_name, model, loader, attack_fn, attack_name, eps_params):
    device = next(model.parameters()).device
    model.eval()

    # directories: data/adv/<dataset>/<attack>/ and data/adv/samples/<dataset>/<attack>/
    adv_dir    = os.path.join('data', 'adv', dataset_name, attack_name)
    sample_dir = os.path.join('data', 'adv', 'samples', dataset_name, attack_name)
    os.makedirs(adv_dir,    exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs('results/res',  exist_ok=True)

    y_true, y_pred = [], []

    for i, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        # generate adversarial examples
        if attack_name == 'deepfool':
            adv = attack_fn(model, data, **eps_params)
        else:
            adv = attack_fn(model, data, target, **eps_params)

        preds = model(adv).argmax(dim=1)
        y_true += target.cpu().tolist()
        y_pred += preds.cpu().tolist()

        # save first 10 pairs of original & adversarial
        for j in range(data.size(0)):
            idx = i * loader.batch_size + j
            if idx < 10:
                o_path = os.path.join(sample_dir, f"orig_{idx}.png")
                a_path = os.path.join(sample_dir, f"adv_{idx}.png")
                utils.save_image(data[j], o_path)
                utils.save_image(adv[j],  a_path)

        # save full batch for adversarial training
        torch.save(adv.cpu(), os.path.join(adv_dir, f"batch_{i}.pt"))

    # compute metrics (suppress zero-division)
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(       y_true, y_pred, average='macro')
    rec  = recall_score(   y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)

    msg = (
        f"{dataset_name} {attack_name} | Params={eps_params} | "
        f"Acc={acc:.4f}, F1={f1:.4f}, Rec={rec:.4f}, Prec={prec:.4f}"
    )

    # output to console + results/res
    print(msg)
    path = os.path.join('results/res', f"{dataset_name}_{attack_name}_results.txt")
    with open(path, 'a') as f:
        f.write(msg + '\n')


if __name__ == '__main__':
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available() else
        'cpu'
    )

    # load ResNet checkpoints
    mnist_model = MNIST_ResNet().to(device)
    mnist_model.load_state_dict(torch.load('mnist_res.pt', map_location=device))
    cifar_model = CIFAR10_ResNet().to(device)
    cifar_model.load_state_dict(torch.load('cifar10_res.pt', map_location=device))

    loaders = {'mnist': mnist_loader, 'cifar10': cifar_loader}
    models  = {'mnist': mnist_model, 'cifar10': cifar_model}

    overshoots = [1]
    for eps in overshoots:
        attacks = [
            ('fgsm',      fgsm_attack,    {'epsilon':0.2}),
            ('pgd',       pgd_attack,     {'epsilon':0.2, 'alpha':0.006, 'iters':10}),
            ('deepfool',  deepfool_attack, {'overshoot':2, 'max_iter':10}),
            ('cw',        cw_l2_attack,   {'c':0.01, 'kappa':0, 'max_iter':10, 'lr':0.09}),
        ]

        # MNIST 60% :
        # ATTACK_PARAMS = {
        #     'fgsm':     {'epsilon': 0.2},
        #     'pgd':      {'epsilon': 0.25, 'alpha': 0.006, 'iters': 10},
        #     'deepfool': {'overshoot': 1.8, 'max_iter': 10},
        #     'cw':       {'c': 0.01, 'kappa': 0, 'max_iter': 10, 'lr': 0.09}
        # }

        for name, fn, params in attacks:
            for ds in ['mnist']:     #, 'cifar10'
                evaluate_and_save_res(ds, models[ds], loaders[ds], fn, name, params)
