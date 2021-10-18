'''ResNet with BinaryConnect'''
from .binary_connect import BinaryConnect, ScaledBinaryConnect
from .resnet56 import ResNet56


def ResNet56_BinaryConnect():
    return BinaryConnect(ResNet56())


def ResNet56_ScaledBinaryConnect():
    return ScaledBinaryConnect(ResNet56())
