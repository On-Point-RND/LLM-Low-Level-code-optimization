import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, a, b, bias):
        return torch.matmul(a, b) + bias

def get_inputs():
    a = torch.randn(1024, 256, dtype=torch.float16)
    b = torch.randn(256, 640, dtype=torch.float16)
    bias = torch.randn(640)
    return [a,b,bias]

def get_init_inputs():
    return []  # No special initialization inputs needed