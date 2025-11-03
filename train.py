import torch
from homa import settings
from homa.models import Resnet
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
    if len(train) < 5000:
        augmented_train = AugmentedDataset(train)
        train = torch.utils.data.ConcatDataset([train, augmented_train])

    test_dataloader = torch.utils.data.DataLoader(
        test, shuffle=False, batch_size=settings("batch_size")
    )

    for i in range(settings("size")):
        model = Resnet(info.num_classes, lr=settings("lr"))
        ensemble = Ensemble(Resnet)
        for epoch in range(settings("epochs")):
            pass
        ensemble.record(model)
