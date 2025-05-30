import torch
import torch.nn.functional as F


def cw_l2_attack(model, images, labels, c=1e-2, kappa=0, max_iter=1000, lr=0.01):
    device = next(model.parameters()).device
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    # arctanh trick: w = arctanh(2*images-1)
    w = torch.atanh((images * 2) - 1)
    w = w.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=lr)

    for step in range(max_iter):
        optimizer.zero_grad()
        adv_images = torch.tanh(w) / 2 + 0.5
        outputs = model(adv_images)
        # Compute f(x)
        one_hot = F.one_hot(labels, num_classes=outputs.shape[1]).float()
        f_loss = torch.clamp((one_hot * outputs).sum(dim=1) - outputs.max(dim=1)[0] + kappa, min=0.0).mean()
        l2_loss = F.mse_loss(adv_images, images)
        loss = l2_loss + c * f_loss
        loss.backward()
        optimizer.step()
    adv_images = torch.tanh(w) / 2 + 0.5
    return adv_images.detach()