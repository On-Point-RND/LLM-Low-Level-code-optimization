# _build_net     : adapted from tp.models.FCN
#   torchphysics/src/torchphysics/models/fcn.py:9  (_construct_FC_layers)
#   torchphysics/src/torchphysics/models/fcn.py:61 (FCN.__init__)
#   differences: accepts plain int dims instead of Space objects;
#   returns nn.Sequential (not Points-wrapped) for KernelBench autograd use
# torchphysics_ref : torchphysics/src/torchphysics/utils/differentialoperators/differentialoperators.py:320 (convective)
# Problem : L1Convective_4d
# Variant : small  (N=128, hidden=32)
# Operator: convective (L320)  —  (v \cdot \nabla)u, self-advection in 4D
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Convective term :math:`(v \cdot \nabla)u` that appears e.g. in material derivatives via autograd (4D).
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

    def forward(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u (torch.Tensor):  The vector or scalar field :math:`u` that is convected and should be differentiated.
            x (torch.Tensor):  Input tensor of shape (N, 4), the spatial variable in which respect u should be differentiated. 
            v (torch.Tensor):  Input tensors of shape (N, 4), the flow vector field :math:`v`. Should have the same dimension as x.

        Returns:
            torch.Tensor:  A vector or scalar (+batch-dimension) Tensor, that contains the convective derivative.
        """       

        with torch.enable_grad():
            x = x.requires_grad_(True)
            u = self.net(x)

            # tp.partial (differentialoperators.py:320) 
            Du_rows = []
            for i in range(u.shape[1]):
                Du_i = []
                for vari in [x]:
                    Du_i.append(
                        torch.autograd.grad(u[..., i].sum(), vari, create_graph=True)[0]
                    )
                Du_rows.append(torch.cat(Du_i, dim=-1))
            jac_x = torch.stack(Du_rows, dim=-2)           
            return torch.bmm(jac_x, v.unsqueeze(dim=2)).squeeze(dim=2) 


def make_torchphysics_ref(model: Model):
    """Return an nn.Module that computes (u\cdot\nabla)u via torchphysics (4D)."""
    import torchphysics as tp

    class _Ref(nn.Module):
        def __init__(self, m):
            super().__init__()
            self._m = m

        def forward(self, v, x):
            with torch.enable_grad():
                x = x.requires_grad_(True)
                u = self._m.net(x)
                return tp.convective(u, v, x)

    return _Ref(model)


N      = 128
hidden = 32


def get_inputs():
    torch.manual_seed(0)
    v = torch.randn(N, 4)
    x = torch.randn(N, 4)
    return [v, x]


def get_init_inputs():
    return []
