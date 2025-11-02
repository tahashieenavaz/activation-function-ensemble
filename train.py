from homa import settings
from homa.models import StochasticResnet

from datasets import BG
from datasets import AugmentedDataset

print(BG())
for i in range(settings("size")):
    model = StochasticResnet(3)

    for epoch in range(settings("epochs")):
        pass
