from homa import settings
from homa.models import StochasticResnet
from homa.ensemble import Ensemble

from datasets import BG
from datasets import AugmentedDataset

print(BG())
for i in range(settings("size")):
    ensemble = Ensemble()
    model = StochasticResnet(3)

    for epoch in range(settings("epochs")):
        pass
