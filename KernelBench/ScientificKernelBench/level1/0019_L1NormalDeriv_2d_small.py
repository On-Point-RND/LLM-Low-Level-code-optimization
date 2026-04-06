# _build_net     : adapted from tp.models.FCN
#   torchphysics/src/torchphysics/models/fcn.py:9  (_construct_FC_layers)
#   torchphysics/src/torchphysics/models/fcn.py:61 (FCN.__init__)
#   differences: accepts plain int dims instead of Space objects;
#   returns nn.Sequential (not Points-wrapped) for KernelBench autograd use
# torchphysics_ref : torchphysics/src/torchphysics/utils/differentialoperators/differentialoperators.py:111 (normal_derivative)
# Problem : L1NormalDeriv_2d
# Variant : small  (N=128, hidden=32)
# Operator: normal_derivative (L111)  —  \partial u / \partial n = \nabla u \cdot n
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Normal derivative of a network with respect to the given variable and normal vectors 
    \partial u / \partial normals = \nabla u \cdot normals  via autograd (2D).
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

    def forward(self, x: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor):  Input tensor of shape (N, 2) of the variables in which respect 
            the derivatives have to be computed 
            normals (torch.Tensor): The normal vectors of shape (N, 2) at the points where 
            the derivative has to be computed. In the form: normals = tensor([normal_1, normal_2, ...]
            
        Returns:
            torch.Tensor: \partial u/\partial n tensor of shape (N, 1), where every row contains the values of the normal
            derivatives w.r.t the row of the input variable.
        """
 
        with torch.enable_grad():
            x = x.requires_grad_(True)
            u = self.net(x)
            # tp.normal_derivative (differentialoperators.py:111) which calls tp.grad
            grads = []
            for vari in [x]:
                new_grad = torch.autograd.grad(u.sum(), vari, create_graph=True)[0]
                grads.append(new_grad)
            gradient = torch.column_stack(grads)
            normal_derivatives = gradient * normals
            return normal_derivatives.sum(dim=-1, keepdim=True)
        

def make_torchphysics_ref(model: Model):
    """Return an nn.Module that computes \partial u/\partial n via torchphysics (2D)."""
    import torchphysics as tp

    class _Ref(nn.Module):
        def __init__(self, m):
            super().__init__()
            self._m = m

        def forward(self, x, normals):
            with torch.enable_grad():
                x = x.requires_grad_(True)
                u = self._m.net(x)
                return tp.normal_derivative(u, normals, x)

    return _Ref(model)


N      = 128
hidden = 32


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, 2)
    normals = F.normalize(torch.randn(N, 2), dim=-1)
    return [x, normals]


def get_init_inputs():
    return []
