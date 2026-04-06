# _build_net     : adapted from tp.models.FCN
#   torchphysics/src/torchphysics/models/fcn.py:9  (_construct_FC_layers)
#   torchphysics/src/torchphysics/models/fcn.py:61 (FCN.__init__)
#   differences: accepts plain int dims instead of Space objects;
#   returns nn.Sequential (not Points-wrapped) for functional torch.func use
# torchphysics_ref : torchphysics/src/torchphysics/utils/differentialoperators/differentialoperators.py:233 (jac)
# Problem : L1Jac_4d
# Variant : small  (N=128, hidden=32)
# Operator: jac (L233)  —  J(u)(x1,x2,x3,x4), u:(N,4)
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
    Jacobian J(u) of a vector-valued network output via torch.func (4D).
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
        self.net = self._build_net(4, 32, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net_single = _make_functional_net(self.net)
        return vmap(jacrev(net_single))(x)


def make_torchphysics_ref(model: Model):
    import torchphysics as tp

    class _Ref(nn.Module):
        def __init__(self, m):
            super().__init__()
            self._m = m

        def forward(self, x):
            with torch.enable_grad():
                x = x.requires_grad_(True)
                u = self._m.net(x)
                return tp.jac(u, x)

    return _Ref(model)


N      = 128
hidden = 32


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, 4)
    return [x]


def get_init_inputs():
    return []
