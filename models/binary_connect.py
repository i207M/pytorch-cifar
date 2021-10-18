'''Wrapper for BinaryConnect'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryConnectWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        # collect weights
        self.saved_params = []
        self.target_params = []
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                self.saved_params.append(layer.weight.clone())
                self.target_params.append(layer.weight)
        self.n_weight = len(self.target_params)

    def forward(self, x):
        return self.model(x)

    # deterministic
    def binarize(self):
        self.save()
        for p in self.target_params:
            p.data.copy_(p.data.sign())

    def clip(self):
        clipper = nn.Hardtanh(-1, 1)
        for p in self.target_params:
            p.data.copy_(clipper(p.data))

    def save(self):
        for i in range(self.n_weight):
            self.saved_params[i].copy_(self.target_params[i].data)

    def restore(self):
        for i in range(self.n_weight):
            self.target_params[i].data.copy_(self.saved_params[i])


if __name__ == '__main__':
    bc = BinaryConnectWrapper(nn.Module())
    print(list(bc.named_children()))
