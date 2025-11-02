import mat73
from scipy.io import loadmat
from pathlib import Path

BASE_DATA_DIR = Path(__file__).parent.parent.parent / "datasets"


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
