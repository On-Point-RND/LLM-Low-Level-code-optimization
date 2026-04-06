# _build_net     : adapted from tp.models.FCN
#   torchphysics/src/torchphysics/models/fcn.py:9  (_construct_FC_layers)
#   torchphysics/src/torchphysics/models/fcn.py:61 (FCN.__init__)
#   differences: accepts plain int dims instead of Space objects;
#   returns nn.Sequential (not Points-wrapped) for KernelBench autograd use
# torchphysics_ref : torchphysics/src/torchphysics/utils/differentialoperators/differentialoperators.py:261 (rot)
# Problem : L1Rot_3d
# Variant : small  (N=128, hidden=32)
# Operator: rot (L261)  —  \nabla 	imes u(x,y,z), u:(N,3)
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Rotation / curl \nabla 	imes u of a 3-dimensional vector field  (given by a network output) with respect to 
    the given input via autograd (3D).
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
        self.net = self._build_net(3, 32, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor):  Input tensor of shape (N, 3) in which respect the rotation should be
        computed.
            
        Returns:
            torch.Tensor: curl tensor of shape (N, 3), where every row contains a rotation/curl vector 
            for a given batch element.
        """      
 
        with torch.enable_grad():
            x = x.requires_grad_(True)
            u = self.net(x)

            Du_rows = []
            for i in range(u.shape[1]):
                Du_i = []
                for vari in [x]:
                    Du_i.append(
                        torch.autograd.grad(u[..., i].sum(), vari, create_graph=True)[0]
                    )
                Du_rows.append(torch.cat(Du_i, dim=-1))
            jacobian = torch.stack(Du_rows, dim=-2)

            rotation = torch.zeros((*(jacobian.shape[:-2]), 3))
            rotation[..., 0] = jacobian[..., 2, 1] - jacobian[..., 1, 2]
            rotation[..., 1] = jacobian[..., 0, 2] - jacobian[..., 2, 0]
            rotation[..., 2] = jacobian[..., 1, 0] - jacobian[..., 0, 1]
            return rotation


def make_torchphysics_ref(model: Model):
    """Return an nn.Module that computes \nabla	imes u via torchphysics (3D)."""
    import torchphysics as tp

    class _Ref(nn.Module):
        def __init__(self, m):
            super().__init__()
            self._m = m

        def forward(self, x):
            with torch.enable_grad():
                x = x.requires_grad_(True)
                u = self._m.net(x)
                return tp.rot(u, x)

    return _Ref(model)


N      = 128
hidden = 32


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, 3)
    return [x]


def get_init_inputs():
    return []
