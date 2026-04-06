# _build_net     : adapted from tp.models.FCN
#   torchphysics/src/torchphysics/models/fcn.py:9  (_construct_FC_layers)
#   torchphysics/src/torchphysics/models/fcn.py:61 (FCN.__init__)
#   differences: accepts plain int dims instead of Space objects;
#   returns nn.Sequential (not Points-wrapped) for KernelBench autograd use
# torchphysics_ref : torchphysics/src/torchphysics/utils/differentialoperators/differentialoperators.py:11 (laplacian)
# Problem : L1Laplacian_4d
# Variant : large  (N=1024, hidden=128)
# Operator: laplacian (L11)  —  \Delta u(x1,x2,x3,x4) = 0
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

def _laplacian(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """\Delta u w.r.t. x via autograd (mirrors tp.utils.laplacian)."""
    laplacian = torch.zeros((*u.shape[:-1], 1), device=u.device)
    for vari in [x]:
        g = torch.autograd.grad(u.sum(), vari, create_graph=True)[0]
        if g.grad_fn is None:
            continue
        for i in range(vari.shape[-1]):
            D2u = torch.autograd.grad(
                g.narrow(-1, i, 1).sum(), vari, create_graph=True
            )[0]
            laplacian += D2u.narrow(-1, i, 1)
    return laplacian


class Model(nn.Module):
    """
    Laplacian of a network \Delta u with respect to the given variable (4D).
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
        self.net = self._build_net(4, 128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor):  Input tensor of shape (N, 4) of the variables in which respect 
            the derivatives have to be computed 
            
        Returns:
            torch.Tensor: \Delta u tensor of shape (N, 1), where every row contains the value of the sum 
            of the second derivatives (laplace) w.r.t the row of the input variable
        """  

        with torch.enable_grad():
            x = x.requires_grad_(True)
            u = self.net(x)
            # tp.laplacian (differentialoperators.py:11) 
            return _laplacian(u, x)     


def make_torchphysics_ref(model: Model):
    """Return an nn.Module that computes \Delta u via torchphysics (4D)."""
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
    x = torch.randn(N, 4)
    return [x]


def get_init_inputs():
    return []
