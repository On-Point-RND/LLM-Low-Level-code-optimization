# _build_net     : adapted from tp.models.FCN
#   torchphysics/src/torchphysics/models/fcn.py:9  (_construct_FC_layers)
#   torchphysics/src/torchphysics/models/fcn.py:61 (FCN.__init__)
#   differences: accepts plain int dims instead of Space objects;
#   returns nn.Sequential (not Points-wrapped) for KernelBench autograd use
# torchphysics_ref : torchphysics/src/torchphysics/utils/differentialoperators/differentialoperators.py:294 (partial)
# Problem : L1Partial_3dt
# Variant : small  (N=128, hidden=32)
# Operator: partial (L294)  —  \partial u/\partial t, spatial dim=3
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    The (n-th, possibly mixed) \partial u / \partial t partial derivative of a network output with
    respect to the given variables via autograd (3D+t).
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
        self.net = self._build_net(4, 32, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor):  Input tensor of shape (N, 3)
            t (torch.Tensor):  Input tensors of shape (N, 1) in which respect the derivatives should be computed. If n
        tensors are given, the n-th (mixed) derivative will be computed.
            
        Returns:
            torch.Tensor: tensor \partial u/\partial t of shape (N, 1), where every row contains the values 
            of the computed partial derivative of the model w.r.t the row of the input variable.
        """       

        with torch.enable_grad():
            t = t.requires_grad_(True)
            xt = torch.cat([x, t], dim=-1)
            u = self.net(xt)

            # tp.partial (differentialoperators.py:294) 
            du = u
            for inp in [t]:
                if du.grad_fn is None:
                    return torch.zeros_like(inp)
                du = torch.autograd.grad(du.sum(), inp, create_graph=True)[0]
            return du       


def make_torchphysics_ref(model: Model):
    """Return an nn.Module that computes \partial u / \partial t via torchphysics (3D+t)."""
    import torchphysics as tp

    class _Ref(nn.Module):
        def __init__(self, m):
            super().__init__()
            self._m = m

        def forward(self, x, t):
            with torch.enable_grad():
                t = t.requires_grad_(True)
                xt = torch.cat([x, t], dim=-1)
                u = self._m.net(xt)
                return tp.partial(u, t)

    return _Ref(model)


N      = 128
hidden = 32


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, 3)
    t = torch.rand(N, 1)
    return [x, t]


def get_init_inputs():
    return []
