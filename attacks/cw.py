import torch
import torch.nn.functional as F

def cw_l2_attack(model, images, labels, c: float = 1e-2, kappa: float = 0, max_iter: int = 100, lr: float = 0.01):
    device = images.device
    batch_size = images.size(0)

    # 1) map images to unconstrained space via arctanh
    #    x = (images*2 - 1) -> in (-1,1)
    #    scaled by 0.999999 to avoid infinities
    x_tanh = torch.atanh((images * 2 - 1) * 0.999999).detach()

    # 2) set up w variable to optimize
    w = x_tanh.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([w], lr=lr)

    # one-hot encode labels
    with torch.no_grad():
        outputs = model(images)
    num_classes = outputs.shape[1]
    labels_onehot = F.one_hot(labels, num_classes).float().to(device)

    for _ in range(max_iter):
        # reconstruct adversarial images and clamp into [0,1]
        adv_images = torch.tanh(w) * 0.5 + 0.5  # in [0,1]

        # forward
        logits = model(adv_images)

        # 3) classification loss: max( real_logit - other_logit + kappa, 0 )
        real_logits  = (labels_onehot * logits).sum(dim=1)            # (B,)
        # for each sample, the highest logit that's NOT the true class:
        other_logits = ((1 - labels_onehot) * logits - labels_onehot * 1e4) \
                       .max(dim=1)[0]                                # (B,)
        f_loss = torch.clamp(real_logits - other_logits + kappa, min=0)  # (B,)

        # 4) L2 distance loss
        l2_loss = torch.sum((adv_images - images) ** 2, dim=(1,2,3))   # (B,)

        # 5) total
        loss = torch.sum(c * l2_loss + f_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # final adversarial examples
    adv_images = torch.tanh(w) * 0.5 + 0.5
    return adv_images.detach()
