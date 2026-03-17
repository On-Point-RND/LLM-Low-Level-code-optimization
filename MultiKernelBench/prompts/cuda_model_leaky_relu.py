import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, negative_slope: float = 0.1):
        super(Model, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)

def get_inputs():
    x = torch.randn(8, 200, 1024)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed