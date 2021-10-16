import torch

from models.resnet56 import ResNet56
from utils import count_params


def test():
    net = ResNet56()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
    count_params(net)


test()
