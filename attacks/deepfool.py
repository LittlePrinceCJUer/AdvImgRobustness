import torch
import torch.nn.functional as F

def deepfool_attack(model, images, num_classes=10, overshoot=0.02, max_iter=10):
    """
    Vectorized DeepFool on a batch of images.
    images: (B,C,H,W), values in [0,1]
    Returns adversarial images of same shape.
    """
    device = next(model.parameters()).device
    images = images.clone().detach().to(device)
    batch_size = images.size(0)

    # Start from the clean images
    pert = images.clone().detach().requires_grad_(True)
    # Initialize accumulated perturbation
    r_tot = torch.zeros_like(images, device=device)

    # Get the original labels
    outputs = model(pert)
    orig_labels = outputs.argmax(dim=1)

    for iter_idx in range(max_iter):
        outputs = model(pert)            # (B, num_classes)
        f_orig = outputs.gather(1, orig_labels.unsqueeze(1)).squeeze(1)  # (B,)
        grad_orig = torch.autograd.grad(f_orig.sum(), pert, retain_graph=True)[0]  # (B,C,H,W)

        # compute gradients and scores for all classes
        grads_k, f_ks = [], []
        for k in range(num_classes):
            f_k = outputs[:, k]           # (B,)
            grad_k = torch.autograd.grad(f_k.sum(), pert, retain_graph=True)[0]  # (B,C,H,W)
            grads_k.append(grad_k)
            f_ks.append(f_k)

        grads_k = torch.stack(grads_k, dim=0)  # (num_classes, B, C, H, W)
        f_ks    = torch.stack(f_ks,    dim=0)  # (num_classes, B)

        # compute distance for each class and each sample
        w_diff = grads_k - grad_orig.unsqueeze(0)     # (num_classes, B, C, H, W)
        f_diff = f_ks    - f_orig.unsqueeze(0)        # (num_classes, B)

        # ||w_diff|| per class/sample
        w_norm = w_diff.view(w_diff.size(0), batch_size, -1).norm(dim=2)      # (num_classes, B)
        dists  = (f_diff.abs() + 1e-8) / (w_norm + 1e-8)                     # (num_classes, B)

        # mask out the original class distances
        mask = torch.arange(num_classes, device=device).unsqueeze(1) == orig_labels.unsqueeze(0)
        dists = dists.masked_fill(mask, float('inf'))

        # pick the minimal‐distance class for each sample
        _, k_i = dists.min(dim=0)  # (B,)

        # gather per‐sample w_i and f_i
        w_i = w_diff[k_i, torch.arange(batch_size)]        # (B,C,H,W)
        f_i = f_diff[k_i, torch.arange(batch_size)]        # (B,)

        # compute per‐sample perturbation
        # note broadcasting from (B,1,1,1)
        r_i = ((f_i.abs() + 1e-4) / (w_i.view(batch_size, -1).norm(dim=1)**2 + 1e-8)
              ).view(batch_size, 1, 1, 1) * w_i            # (B,C,H,W)

        # accumulate and apply
        r_tot = r_tot + r_i
        pert  = torch.clamp(images + (1 + overshoot) * r_tot, 0.0, 1.0
               ).detach().requires_grad_(True)

        # early stop once any label flips
        new_labels = model(pert).argmax(dim=1)
        if not torch.equal(new_labels, orig_labels):
            break

    return pert.detach()
