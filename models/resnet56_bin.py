'''ResNet with BinaryConnect'''
from binary_connect import BinaryConnectWrapper
from resnet56 import ResNet56


def ResNet56_bin():
    return BinaryConnectWrapper(ResNet56())
