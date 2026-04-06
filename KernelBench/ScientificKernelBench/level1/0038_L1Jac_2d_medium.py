# _build_net     : adapted from tp.models.FCN
#   torchphysics/src/torchphysics/models/fcn.py:9  (_construct_FC_layers)
#   torchphysics/src/torchphysics/models/fcn.py:61 (FCN.__init__)
#   differences: accepts plain int dims instead of Space objects;
#   returns nn.Sequential (not Points-wrapped) for KernelBench autograd use
# torchphysics_ref : torchphysics/src/torchphysics/utils/differentialoperators/differentialoperators.py:233 (jac)
# Problem : L1Jac_2d
# Variant : medium  (N=512, hidden=64)
# Operator: jac (L233)  —  J(u)(x,y), u:(N,2)
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


def _jac(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    Du_rows = []
    for i in range(u.shape[1]):
        Du_i = []
        for vari in [x]:
            Du_i.append(
                torch.autograd.grad(u[..., i].sum(), vari, create_graph=True)[0]
            )
        Du_rows.append(torch.cat(Du_i, dim=-1))
    Du = torch.stack(Du_rows, dim=-2)
    return Du


class Model(nn.Module):
    """
    Jacobian  J(u) of a network output with respect to the given input via autograd (2D).
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
        self.net = self._build_net(2, 64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor):  Input tensor of shape (N, 2) in which respect the jacobian should be computed.
            
        Returns:
            torch.Tensor:  J(u) tensor of shape (N, 2, 2), where every row contains a jacobian.
        """       

        with torch.enable_grad():
            x = x.requires_grad_(True)
            u = self.net(x)

            # tp.jac (differentialoperators.py:233) 
            return _jac(u, x)


def make_torchphysics_ref(model: Model):
    """Return an nn.Module that computes J(u) via torchphysics (2D)."""
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


N      = 512
hidden = 64


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, 2)
    return [x]


def get_init_inputs():
    return []
