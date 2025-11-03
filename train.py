from homa import settings
from homa.models import StochasticResnet, ResnetWrapper
from homa.ensemble import Ensemble

from datasets import BG
from datasets import AugmentedDataset

info = BG()
for idx, fold in enumerate(info.folds):
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
