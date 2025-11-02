import torch
from torch.utils.data import Dataset
from collections.abc import Callable
from torchvision.transforms import v2


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
