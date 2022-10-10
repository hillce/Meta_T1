import torch

from models import Braided_UNet_Complete

net = Braided_UNet_Complete(7,7,1,1)

weights = net.state_dict()

print("{:.2f} KB".format(weights.__sizeof__()/(1024)))




