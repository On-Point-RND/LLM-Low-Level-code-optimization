from torch_xla.experimental.custom_kernel import jax_import_guard
jax_import_guard()
from torch_xla.experimental.custom_kernel import make_kernel_from_pallas
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import torch
import torch.nn as nn

# Define the custom Pallas kernel for element-wise addition
def elementwise_add_kernel(x_ref, y_ref, z_ref):
    x = x_ref[...]  # Load full tensor using memory reference
    y = y_ref[...]
    z_ref[...] = x + y  # Store result

@jax.jit
def elementwise_add_pallas(x: jax.Array, y: jax.Array) -> jax.Array:
    return pl.pallas_call(
        elementwise_add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x, y)

# Compile and create PyTorch compatible kernel
pt_kernel = make_kernel_from_pallas(
    elementwise_add_pallas,
    lambda x, y: [(x.shape, x.dtype)]  # Shape inference function
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return pt_kernel(a, b)