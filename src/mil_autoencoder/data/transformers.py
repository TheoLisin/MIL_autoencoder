from torchvision import transforms

from mil_autoencoder.utils.autoaugment import CIFAR10Policy


def cifar_augment_transform():
    augment_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return augment_transform


def simple_transform():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
    )
    return transform
