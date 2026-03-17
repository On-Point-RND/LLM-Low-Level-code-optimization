import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs sum reduction to one dimention vector.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x)

def get_inputs():
    x = torch.randn(4096)
    return [x]

def get_init_inputs():
    return []