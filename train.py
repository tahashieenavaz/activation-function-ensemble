import torch
from homa import settings
from homa.models import StochasticResnet, ResnetWrapper
from homa.ensemble import Ensemble
from types import SimpleNamespace
from datasets import BG
from datasets import AugmentedDataset

info = SimpleNamespace(**BG())
for idx, fold in enumerate(info.folds):
    fold = list(map(lambda x: x - 1, fold.astype(int)))
    train_idx = fold[: info.threshold]
    test_idx = fold[info.threshold :]
    train = torch.utils.data.Subset(info.dataset, train_idx)
    test = torch.utils.data.Subset(info.dataset, test_idx)

    for i in range(settings("size")):
        model = ResnetWrapper(
            architecture=StochasticResnet,
            num_classes=info.num_classes,
            lr=settings("lr"),
        )
        ensemble = Ensemble(num_classes=info.num_classes)
        for epoch in range(settings("epochs")):
            pass
        ensemble.record(model)
