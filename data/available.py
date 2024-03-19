from torchvision import datasets, transforms
from data.manipulate import UnNormalize


class ImageNet100(datasets.ImageFolder):
    def __init__(self, root, train=False, transform=None, target_transform=None, download=False):
        if train:
            root = root + "/train"
        else:
            root = root + "/val"
        super().__init__(root, transform=transform, target_transform=target_transform)


# Specify available data-sets
AVAILABLE_DATASETS = {
    "mnist": datasets.MNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "imagenet100": ImageNet100,
}


# Specify available transforms
AVAILABLE_TRANSFORMS = {
    "mnist": [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    "mnist28": [
        transforms.ToTensor(),
    ],
    "cifar10": [
        transforms.ToTensor(),
    ],
    "cifar100": [
        transforms.ToTensor(),
    ],
    "imagenet100": [
        transforms.Resize((76, 76)),
        transforms.CenterCrop((64, 64)),
        transforms.ToTensor(),
    ],
    "imagenet100_norm": [
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ],
    "cifar10_norm": [
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
    ],
    "cifar100_norm": [
        transforms.Normalize(
            mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]
        )
    ],
    "cifar10_denorm": UnNormalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
    ),
    "cifar100_denorm": UnNormalize(
        mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]
    ),
    "imagenet100_denorm": UnNormalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    ),
    "augment_from_tensor": [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4, padding_mode="symmetric"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ],
    "augment": [
        # transforms.RandomCrop(32, padding=4, padding_mode="symmetric"),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.RandomErasing(scale=(0.1, 0.5)),
    ],
}


# Specify configurations of available data-sets
DATASET_CONFIGS = {
    "mnist": {"size": 32, "channels": 1, "classes": 10},
    "mnist28": {"size": 28, "channels": 1, "classes": 10},
    "cifar10": {"size": 32, "channels": 3, "classes": 10},
    "cifar100": {"size": 32, "channels": 3, "classes": 100},
    "imagenet100": {"size": 64, "channels": 3, "classes": 100},
}
