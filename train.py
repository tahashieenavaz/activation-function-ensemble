import torch
from homa import settings
from homa.vision import Resnet, StochasticResnet, Swin, StochasticSwin
from homa.ensemble import Ensemble
from copy import deepcopy
from types import SimpleNamespace
from datasets import BG
from datasets import AugmentedDataset

info = SimpleNamespace(**BG())
for fold_idx, fold in enumerate(info.folds):
    fold = list(map(lambda x: x - 1, fold.astype(int)))
    train_idx = fold[: info.threshold]
    test_idx = fold[info.threshold :]
    train = torch.utils.data.Subset(info.dataset, train_idx)
    test = torch.utils.data.Subset(info.dataset, test_idx)
    if len(train) < 5000:
        train = torch.utils.data.ConcatDataset([train, AugmentedDataset(train)])

    test_dataloader = torch.utils.data.DataLoader(
        test, shuffle=False, batch_size=settings("batch_size")
    )

    print(f"fold: {fold_idx + 1}", flush=True)
    ensemble = Ensemble()
    for i in range(settings("size")):
        print(f"\tmodel: {i + 1}")
        model = StochasticSwin(num_classes=info.num_classes)
        best_model, best_accuracy = None, float("-inf")
        for epoch in range(settings("epochs")):
            train_dataloader = torch.utils.data.DataLoader(
                train, shuffle=True, batch_size=settings("batch_size")
            )
            model.train(train_dataloader)
            accuracy = model.accuracy(test_dataloader)
            if accuracy > best_accuracy:
                best_model = deepcopy(model.network)
                best_accuracy = accuracy
            print(f"\t\tepoch: {epoch + 1}, accuracy: {accuracy}", flush=True)
        print(f"\t\trecorded model accuracy: {best_accuracy}")
        ensemble.record(best_model)
    print(f"Ensemble accuracy: {ensemble.accuracy(test_dataloader)}")
