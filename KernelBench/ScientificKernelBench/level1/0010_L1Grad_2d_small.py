# _build_net     : adapted from tp.models.FCN
#   torchphysics/src/torchphysics/models/fcn.py:9  (_construct_FC_layers)
#   torchphysics/src/torchphysics/models/fcn.py:61 (FCN.__init__)
#   differences: accepts plain int dims instead of Space objects;
#   returns nn.Sequential (not Points-wrapped) for KernelBench autograd use
# torchphysics_ref : torchphysics/src/torchphysics/utils/differentialoperators/differentialoperators.py:47 (grad)
# Problem : L1Grad_2d
# Variant : small  (N=128, hidden=32)
# Operator: grad (L47)  —  \nabla u(x,y)
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Gradient of a network \nabla u with respect to the given variable (2D).
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
        self.net = self._build_net(2, 32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor):  Input tensor of shape (N, 2) of the variables in which respect 
            the derivatives have to be computed
            
        Returns:
            torch.Tensor: \nabla u tensor of shape (N, 2), where every row contains the values of 
            the first derivatives (gradient) w.r.t the row of the input variable 
        """
        
        with torch.enable_grad():
            x = x.requires_grad_(True)
            u = self.net(x)
            #  tp.grad (differentialoperators.py:47) 
            grads = []
            for vari in [x]:
                new_grad = torch.autograd.grad(u.sum(), vari, create_graph=True)[0]
                grads.append(new_grad)
            return torch.column_stack(grads)


def make_torchphysics_ref(model: Model):
    """Return an nn.Module that computes \nabla u via torchphysics (2D)."""
    import torchphysics as tp

    class _Ref(nn.Module):
        def __init__(self, m):
            super().__init__()
            self._m = m

        def forward(self, x):
            with torch.enable_grad():
                x = x.requires_grad_(True)
                u = self._m.net(x)
                return tp.grad(u, x)

    return _Ref(model)


N      = 128
hidden = 32


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, 2)
    return [x]


def get_init_inputs():
    return []
