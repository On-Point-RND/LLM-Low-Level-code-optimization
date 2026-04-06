# _build_net     : adapted from tp.models.FCN
#   torchphysics/src/torchphysics/models/fcn.py:9  (_construct_FC_layers)
#   torchphysics/src/torchphysics/models/fcn.py:61 (FCN.__init__)
#   differences: accepts plain int dims instead of Space objects;
#   returns nn.Sequential (not Points-wrapped) for functional torch.func use
# torchphysics_ref : torchphysics/src/torchphysics/utils/differentialoperators/differentialoperators.py:11 (laplacian)
# Problem : L1Laplacian_2d
# Variant : large  (N=1024, hidden=128)
# Operator: laplacian (L11)  —  \Delta u(x,y) = 0
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.func import functional_call, vmap, jacrev, hessian


def _make_functional_net(net: nn.Module):
    params = dict(net.named_parameters())
    buffers = dict(net.named_buffers())

    def net_single(x_single: torch.Tensor) -> torch.Tensor:
        out = functional_call(net, (params, buffers), (x_single.unsqueeze(0),))
        return out.squeeze(0)

    return net_single


def _scalarize_last_dim(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 1 and y.shape[0] == 1:
        return y[0]
    return y.squeeze()


def _trace_last_two(mat: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(mat, dim1=-2, dim2=-1).sum(dim=-1)


def _symmetrize_last_two(mat: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mat + mat.transpose(-2, -1))


class Model(nn.Module):
    """
    Laplacian of a network \Delta u with respect to the given variable (2D),
    implemented with torch.func for compiler-friendly higher-order derivatives.
    """

    @staticmethod
    def _build_net(in_dim, hidden, out_dim):
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, out_dim)]
        net = nn.Sequential(*layers)
        torch.manual_seed(42)
        for p in net.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        return net

    def __init__(self):
        super().__init__()
        self.net = self._build_net(2, 128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net_single = _make_functional_net(self.net)

        def scalar_field(x_single: torch.Tensor) -> torch.Tensor:
            return _scalarize_last_dim(net_single(x_single))

        H = vmap(hessian(scalar_field))(x)
        lap = _trace_last_two(H).unsqueeze(-1)
        return lap


def make_torchphysics_ref(model: Model):
    """Return an nn.Module that computes \Delta u via torchphysics (2D)."""
    import torchphysics as tp

    class _Ref(nn.Module):
        def __init__(self, m):
            super().__init__()
            self._m = m

        def forward(self, x):
            with torch.enable_grad():
                x = x.requires_grad_(True)
                u = self._m.net(x)
                return tp.laplacian(u, x)

    return _Ref(model)


N      = 1024
hidden = 128


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, 2)
    return [x]


def get_init_inputs():
    return []
