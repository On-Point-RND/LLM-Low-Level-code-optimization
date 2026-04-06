# _build_net     : adapted from tp.models.FCN
#   torchphysics/src/torchphysics/models/fcn.py:9  (_construct_FC_layers)
#   torchphysics/src/torchphysics/models/fcn.py:61 (FCN.__init__)
#   differences: accepts plain int dims instead of Space objects;
#   returns nn.Sequential (not Points-wrapped) for KernelBench autograd use
# torchphysics_ref : torchphysics/src/torchphysics/utils/differentialoperators/differentialoperators.py:137 (div)
# Problem : L1Div_4d
# Variant : small  (N=128, hidden=32)
# Operator: div (L137)  —  \nabla \cdot u(x1,x2,x3,x4)
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


def _div(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    divergence = torch.zeros((*x.shape[:-1], 1), device=x.device)
    var_dim = 0
    for vari in [x]:
        for i in range(vari.shape[-1]):
            Du = torch.autograd.grad(
                u.narrow(-1, var_dim + i, 1).sum(), vari, create_graph=True
            )[0]
            divergence = divergence + Du.narrow(-1, i, 1)
        var_dim += i + 1
    return divergence


class Model(nn.Module):
    """
    Divergence of a network with respect to the given variable. Only for vector valued inputs, 
    for matices use the function matrix_div. \nabla \cdot u  via autograd (4D).
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
        """
        Args:
            x (torch.Tensor):  Input tensor of shape (N, 4) of the variables in which respect 
            the derivatives have to be computed. Have to be in a consistent ordering, if for example 
            the output is u = (u_x, u_y) than the variables has to passed in the order (x, y)
            
        Returns:
            torch.Tensor: \nabla\cdot u tensor of shape (N, 1), where every row contains the values 
            of the divergence of the model w.r.t the row of the input variable.
        """    

        with torch.enable_grad():
            x = x.requires_grad_(True)
            u = self.net(x)

            # tp.div (differentialoperators.py:137) 
            return _div(u, x)


def make_torchphysics_ref(model: Model):
    """Return an nn.Module that computes \nabla\cdot u via torchphysics (4D)."""
    import torchphysics as tp

    class _Ref(nn.Module):
        def __init__(self, m):
            super().__init__()
            self._m = m

        def forward(self, x):
            with torch.enable_grad():
                x = x.requires_grad_(True)
                u = self._m.net(x)
                return tp.div(u, x)

    return _Ref(model)


N      = 128
hidden = 32


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, 4)
    return [x]


def get_init_inputs():
    return []
