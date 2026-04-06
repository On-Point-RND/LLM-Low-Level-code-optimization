# _build_net     : adapted from tp.models.FCN
#   torchphysics/src/torchphysics/models/fcn.py:9  (_construct_FC_layers)
#   torchphysics/src/torchphysics/models/fcn.py:61 (FCN.__init__)
#   differences: accepts plain int dims instead of Space objects;
#   returns nn.Sequential (not Points-wrapped) for KernelBench autograd use
# torchphysics_ref : torchphysics/src/torchphysics/utils/differentialoperators/differentialoperators.py:365 (matrix_div)
# Problem : L1MatrixDiv_4d
# Variant : large  (N=1024, hidden=128)
# Operator: matrix_div (L365)  —  \nabla \cdot \sigma, \sigma:(N,4,4) in 4D
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Matrix divergence  \nabla \cdot \sigma  via autograd (4D).

    Net outputs \sigma_flat:(N,16) reshaped to \sigma:(N,4,4).
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
        self.net = self._build_net(4, 128, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u (torch.Tensor):  The (batch) of matirces that should be differentiated.
            x (torch.Tensor):  Input tensor of shape (N, 4), the spatial variable in which respect u should be differentiated. 

        Returns:
            torch.Tensor:  A Tensor of vectors of the form (N, dim), containing the
            divegrence of the input.
        """      
        with torch.enable_grad():
            x = x.requires_grad_(True)
            sigma = self.net(x).view(-1, 4, 4)

            # tp.partial (differentialoperators.py:365) 
            '''
            div_out = torch.zeros(x.shape[0], 4, device=x.device, dtype=x.dtype)
            for i in range(4):
                for j in range(4):
                    g = torch.autograd.grad(
                        sigma[:, i, j].sum(), x, create_graph=True
                    )[0]
                    div_out[:, i] = div_out[:, i] + g[:, j]
            return div_out
            '''

            div_out = torch.zeros((len(sigma), sigma.shape[1]), device=sigma.device)
            for i in range(sigma.shape[1]):
                # compute divergence of matrix by computing the divergence
                # for each row
                current_row = sigma.narrow(1, i, 1).squeeze(1)

                # tp.div (differentialoperators.py:137) 
                divergence = torch.zeros((*x.shape[:-1], 1), device=x.device)
                var_dim = 0
                for vari in [x]:
                    for j in range(vari.shape[-1]):
                        Du = torch.autograd.grad(
                            current_row.narrow(-1, var_dim + j, 1).sum(), vari, create_graph=True
                        )[0]
                        divergence = divergence + Du.narrow(-1, j, 1)
                    var_dim += j + 1

                div_out[:, i : i + 1] = divergence    
            return div_out            


def make_torchphysics_ref(model: Model):
    """Return an nn.Module that computes \nabla\cdot\sigma via torchphysics (4D)."""
    import torchphysics as tp

    class _Ref(nn.Module):
        def __init__(self, m):
            super().__init__()
            self._m = m

        def forward(self, x):
            with torch.enable_grad():
                x = x.requires_grad_(True)
                sigma = self._m.net(x).view(-1, 4, 4)
                return tp.matrix_div(sigma, x)

    return _Ref(model)


N      = 1024
hidden = 128


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, 4)
    return [x]


def get_init_inputs():
    return []
