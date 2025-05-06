import torchvision.transforms as transforms


def get_mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),  # min max transformation: [0,255] to [0.0,1.0]
    ])


def get_cifar10_transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to 64 * 64
        # Randomly crop and resize to introduce scale variation
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        # Random horizontal and vertical flips to augment orientation
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # Small rotations for additional robustness
        transforms.RandomRotation(15),
        # Convert to tensor (min-max to [0,1])
        transforms.ToTensor(),
    ])