import torch
import torch.nn.functional as F

def fgsm_attack(model, images, labels, epsilon):
    images = images.clone().detach().to(next(model.parameters()).device)
    labels = labels.to(next(model.parameters()).device)
    images.requires_grad = True

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()

    # Collect sign of gradient
    sign_data_grad = images.grad.sign()
    # Create perturbed image by adjusting each pixel
    perturbed_images = images + epsilon * sign_data_grad
    # Clamp to [0,1]
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images