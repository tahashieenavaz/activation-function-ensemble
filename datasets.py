import torch
import mat73
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2
from collections.abc import Callable
from pathlib import Path

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BASE_DATA_DIR = Path(__file__).parent.parent / "datasets"


class ImageDataset(Dataset):
    def __init__(self, images: list, labels: list, transform: Callable | None = None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class AugmentedDataset(Dataset):
    def __init__(self, original: Dataset, pipeline: Callable = v2.RandAugment()):
        super().__init__()
        self.original = original
        self.pipeline = pipeline

    def __len__(self) -> int:
        return len(self.original)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.original[idx]
        return self.pipeline(image), label


def build_transforms(is_grayscale: bool = False) -> transforms.Compose:
    transform_list = [
        transforms.ToPILImage(),
    ]

    # If image is grayscale, convert it to 3 channels for pretrained models
    if is_grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=3))

    transform_list += [
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(transform_list)


def _load_scipy_dataset(filename: str) -> list:
    filepath = BASE_DATA_DIR / filename
    data = loadmat(filepath)["DATA"][0]
    images = data[0][0]
    labels = data[1][0]
    folds = data[2]
    threshold = data[3][0][0]
    labels = [label - 1 for label in labels]
    is_grayscale = len(images[0].shape) == 2
    transform = build_transforms(is_grayscale=is_grayscale)

    return [
        ImageDataset(images=images, labels=labels, transform=transform),
        len(set(labels)),
        folds,
        threshold,
    ]


def _load_mat73_dataset(filename: str) -> list:
    filepath = BASE_DATA_DIR / filename
    data = mat73.loadmat(filepath)["DATA"]
    images = data[0]
    labels = data[1]
    folds = data[2]
    threshold = data[3]
    labels = [label - 1 for label in labels]
    is_grayscale = len(images[0].shape) == 2
    transform = build_transforms(is_grayscale=is_grayscale)
    return [
        ImageDataset(images=images, labels=labels, transform=transform),
        len(set(labels)),
        folds,
        threshold,
    ]


def BG():
    return _load_scipy_dataset("BG.mat")


def LAR():
    return _load_scipy_dataset("LAR.mat")


def DENG():
    return _load_scipy_dataset("DENG.mat")


def VIR():
    return _load_scipy_dataset("VIR.mat")


def WHOI():
    return _load_mat73_dataset("WHOI.mat")


def KAGGLE():
    return _load_mat73_dataset("KAGGLE.mat")


def ZOOLAKE():
    return _load_mat73_dataset("ZOOLAKE.mat")


def ZOOSCAN():
    return _load_mat73_dataset("ZOOSCAN.mat")
