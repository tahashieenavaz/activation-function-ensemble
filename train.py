from homa import settings
from homa.models import StochasticResnet
from homa.ensemble import Ensemble

from datasets import BG
from datasets import AugmentedDataset

dataset = BG()
for i in range(settings("size")):
    model = StochasticResnet(3)
    ensemble = Ensemble()

    for epoch in range(settings("epochs")):
        pass
