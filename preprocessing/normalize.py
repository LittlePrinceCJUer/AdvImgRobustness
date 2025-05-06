import torchvision.transforms as transforms


def get_mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),  # min max transformation: [0,255] to [0.0,1.0]
    ])


def get_cifar10_transform():
    return transforms.Compose([
        transforms.ToTensor(),  # min max transformation: [0,255] to [0.0,1.0]
    ])