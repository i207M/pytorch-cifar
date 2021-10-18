'''ResNet with BinaryConnect'''
from .binary_connect import BinaryConnectWrapper
from .one_bit import OneBitWrapper
from .resnet56 import ResNet56


def ResNet56_BinaryConnect():
    return BinaryConnectWrapper(ResNet56())


def ResNet56_OneBit():
    return OneBitWrapper(ResNet56())
