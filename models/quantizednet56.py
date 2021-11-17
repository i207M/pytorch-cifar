'''Wrapper for BinaryConnect and Binary Weight Network'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A
from torch.autograd.function import once_differentiable
from torch.functional import Tensor
from torch.nn.parameter import Parameter


class QuantizationFunction(A.Function):
    @staticmethod
    def forward(ctx, input: Tensor, alpha: Tensor) -> Tensor:
        ctx.alpha = alpha.item()
        ctx.quantized = input.sign()
        return ctx.quantized * ctx.alpha

    @once_differentiable
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        grad_input = grad_alpha = None

        if ctx.needs_input_grad[0]:
            grad_input = ctx.alpha * grad_output
        if ctx.needs_input_gard[1]:
            grad_alpha = grad_output.view(-1).dot(ctx.quantized.view(-1))

        return grad_input, grad_alpha


class QLinear(nn.Linear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = Parameter(torch.randn(1))

    def forward(self, input: Tensor) -> Tensor:
        quantized_weight = QuantizationFunction.apply(self.weight, self.alpha)
        return F.linear(input, quantized_weight, self.bias)


class QConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = Parameter(torch.randn(1))

    def forward(self, input: Tensor) -> Tensor:
        quantized_weight = QuantizationFunction.apply(self.weight, self.alpha)
        return self._conv_forward(input, quantized_weight, self.bias)


class QPreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = QConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(QConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class QPreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = QConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = QLinear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def QuantizedNet56():
    return QPreActResNet(QPreActBlock, [9, 9, 9])


if __name__ == '__main__':
    QuantizedNet56()
