from homa import settings

# from homa.models import StochasticResnet
from datasets import BG

print(BG())
for i in range(settings("size")):
    # model = StochasticResnet()

    for epoch in range(settings("epochs")):
        pass
