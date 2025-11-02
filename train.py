from homa import settings
from homa.models import StochasticResnet


for i in range(settings("size")):
    model = StochasticResnet()

    for epoch in range(settings("epochs")):
        pass
