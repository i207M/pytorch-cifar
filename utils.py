import numpy as np


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
