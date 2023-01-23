import typing as tp
import torchvision
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


def load_data(
    path: tp.Union[str, Path],
    train_transform,
    test_transform,
) -> tp.Tuple[Dataset, Dataset, tp.Tuple[str, ...]]:
    trainset = torchvision.datasets.CIFAR10(
        root=path,
        train=True,
        download=True,
        transform=train_transform,
    )

    testset = torchvision.datasets.CIFAR10(
        root=path,
        train=False,
        download=True,
        transform=test_transform,
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return trainset, testset, classes


def create_loaders(
    trainset: Dataset,
    testset: Dataset,
    train_batch: int = 16,
    test_batch: int = 16,
) -> tp.Tuple[DataLoader, DataLoader]:
    trainloader = DataLoader(
        trainset,
        batch_size=train_batch,
        shuffle=True,
    )
    testloader = DataLoader(
        testset,
        batch_size=test_batch,
        shuffle=False,
    )

    return trainloader, testloader
