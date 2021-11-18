import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def count_params(net):
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


def check_nan(x: Tensor) -> bool:
    arr = x.detach().cpu().numpy()
    not_zero = arr.any()
    assert not_zero
    is_nan = np.isnan(arr).any()
    assert not is_nan


def analyse(model: nn.Module):
    n_total = n_positive = 0
    max_abs = avg_abs = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weight = layer.weight
            n_total += weight.nelement()
            n_positive += weight[weight > 0].count_nonzero().item()

            abs_weight = weight.abs()
            max_abs = max(abs_weight.max().item(), max_abs)
            avg_abs += abs_weight.sum().item()

    avg_abs /= n_total
    print(f'{n_total=}, {n_positive=}, {max_abs=}, {avg_abs=}')
