import torch
import torch.nn.functional as F


def pgd_attack(model, images, labels, epsilon, alpha=0.01, iters=40):
    images = images.clone().detach().to(next(model.parameters()).device)
    labels = labels.to(next(model.parameters()).device)
    ori_images = images.clone().detach()

    # Initialize perturbation
    perturbed = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    perturbed = torch.clamp(perturbed, 0, 1)

    for i in range(iters):
        # make perturbed a leaf Tensor with requires_grad
        perturbed = perturbed.clone().detach().requires_grad_(True)
        outputs = model(perturbed)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = perturbed.grad.data

        # PGD step
        perturbed = perturbed + alpha * torch.sign(grad)
        # Projection
        perturbed = torch.max(torch.min(perturbed, ori_images + epsilon), ori_images - epsilon)
        perturbed = torch.clamp(perturbed, 0, 1)
    return perturbed.detach()