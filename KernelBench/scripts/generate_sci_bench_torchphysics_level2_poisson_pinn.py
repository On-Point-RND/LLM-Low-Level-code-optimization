#!/usr/bin/env python3
"""
generate_sci_bench_torchphysics_level2_poisson_pinn.py
=======================================================
Converts two torchphysics PINN notebook examples into benchmark kernel tasks.

Notebook 1 — poisson-equation.ipynb
  Solves the Laplace equation on Circle(r=1) \\ Square([-0.25,0.25]^2):
    Delta u = 0        on interior D
    u = cos(x1)cos(x2) on boundary partial D
  Network: FCN(hidden=(20,20,20)), input dim=2.
  Tasks written to: tasks/sci_bench_torchphysics/level2/poisson_pinn/

Notebook 2 — poisson-with-input-params.ipynb
  Solves a parametric Poisson equation on an L-shaped domain:
    -D Delta u = g(x,F)  on  small_Rect = [0.6,0.8]^2   (source region)
    -D Delta u = 0        on  Omega \\ small_Rect           (homogeneous region)
    u = 0                 on  partial Omega                (Dirichlet BC)
  D in [0.1,1.0], F in [1.0,5.0] are input parameters.
  Network: FCN(hidden=(50,50,50,50,50)), input dim=4 (x1,x2,D,F).
  Tasks written to: tasks/sci_bench_torchphysics/level2/poisson_input_params_pinn/

References
----------
  torchphysics/examples/pinn/poisson-equation.ipynb
  torchphysics/examples/pinn/poisson-with-input-params.ipynb
  tp.utils.laplacian  ->  differentialoperators.py:11 (laplacian)

Run:
    python scripts/generate_sci_bench_torchphysics_level2_poisson_pinn.py
"""

import math
from pathlib import Path

_BUILD_NET_REF = (
    "# _build_net     : adapted from tp.models.FCN\n"
    "#   torchphysics/src/torchphysics/models/fcn.py:9  (_construct_FC_layers)\n"
    "#   torchphysics/src/torchphysics/models/fcn.py:61 (FCN.__init__)\n"
    "#   differences: accepts plain int dims instead of Space objects;\n"
    "#   returns nn.Sequential (not Points-wrapped) for KernelBench autograd use"
)

_DIFFOPS = (
    "torchphysics/src/torchphysics/utils/differentialoperators/"
    "differentialoperators.py"
)

# FCN(hidden=(20,20,20)) — exactly as in poisson-equation.ipynb
_NET_BLOCK_20 = """\
    @staticmethod
    def _build_net(in_dim, out_dim):
        \"\"\"
        Exact implementation of tp.models.FCN(hidden=(20,20,20)).
        Hidden layers: xavier_normal_(gain=5/3).  Output: gain=1.
        from torchphysics/src/torchphysics/models/fcn.py:9
        \"\"\"
        torch.manual_seed(42)
        gain = 5.0 / 3.0
        l0 = nn.Linear(in_dim, 20); nn.init.xavier_normal_(l0.weight, gain=gain)
        l1 = nn.Linear(20, 20); nn.init.xavier_normal_(l1.weight, gain=gain)
        l2 = nn.Linear(20, 20); nn.init.xavier_normal_(l2.weight, gain=gain)
        l3 = nn.Linear(20, out_dim); nn.init.xavier_normal_(l3.weight, gain=1.0)
        return nn.Sequential(l0, nn.Tanh(), l1, nn.Tanh(), l2, nn.Tanh(), l3)
"""

# FCN(hidden=(50,50,50,50,50)) — exactly as in poisson-with-input-params.ipynb
_NET_BLOCK_50 = """\
    @staticmethod
    def _build_net(in_dim, out_dim):
        \"\"\"
        Exact implementation of tp.models.FCN(hidden=(50,50,50,50,50)).
        Hidden layers: xavier_normal_(gain=5/3).  Output: gain=1.
        from torchphysics/src/torchphysics/models/fcn.py:9
        \"\"\"
        torch.manual_seed(42)
        gain = 5.0 / 3.0
        l0 = nn.Linear(in_dim, 50); nn.init.xavier_normal_(l0.weight, gain=gain)
        l1 = nn.Linear(50, 50);     nn.init.xavier_normal_(l1.weight, gain=gain)
        l2 = nn.Linear(50, 50);     nn.init.xavier_normal_(l2.weight, gain=gain)
        l3 = nn.Linear(50, 50);     nn.init.xavier_normal_(l3.weight, gain=gain)
        l4 = nn.Linear(50, 50);     nn.init.xavier_normal_(l4.weight, gain=gain)
        l5 = nn.Linear(50, out_dim); nn.init.xavier_normal_(l5.weight, gain=1.0)
        return nn.Sequential(
            l0, nn.Tanh(), l1, nn.Tanh(), l2, nn.Tanh(),
            l3, nn.Tanh(), l4, nn.Tanh(), l5
        )
"""


_LAPLACIAN_BLOCK = '''\
def _laplacian(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """\\Delta u w.r.t. x via autograd (mirrors tp.utils.laplacian)."""
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
    return laplacian'''

_NORMAL_DERIVATIVE_BLOCK = '''\
def _normal_derivative(u: torch.Tensor, normals: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """n·\\nabla u via autograd (mirrors tp.utils.normal_derivative, differentialoperators.py:111)."""
    grads = []
    for vari in [x]:
        new_grad = torch.autograd.grad(u.sum(), vari, create_graph=True)[0]
        grads.append(new_grad)
    gradient = torch.column_stack(grads)
    normal_derivatives = gradient * normals
    return normal_derivatives.sum(dim=-1, keepdim=True)
'''

_NORMALS_SQUARE_BLOCK = '''\
def _construct_parallelogram(n_pts: int, device):
    """Returns (origin, corner_1, corner_2, dir_1, dir_2) for the square [0,W]x[0,H].
    Mirrors Parallelogram._construct_parallelogram() (parallelogram.py).
    """
    origin  = torch.zeros(n_pts, 2, device=device)
    corner1 = torch.tensor([[W, 0.0]], device=device).expand(n_pts, -1)
    corner2 = torch.tensor([[0.0, H]], device=device).expand(n_pts, -1)
    dir_1   = corner1 - origin
    dir_2   = corner2 - origin
    return origin, corner1, corner2, dir_1, dir_2


def _solve_lgs(points: torch.Tensor, dir_1: torch.Tensor, dir_2: torch.Tensor):
    """Solve  bary_x * dir_1 + bary_y * dir_2 = points  for each row (Cramer\'s rule).
    Mirrors ParallelogramBoundary._solve_lgs() (parallelogram.py).
    """
    det    = dir_1[:, :1] * dir_2[:, 1:] - dir_1[:, 1:] * dir_2[:, :1]
    bary_x = torch.divide(dir_2[:, 1:] * points[:, :1] - dir_2[:, :1] * points[:, 1:], det)
    bary_y = torch.divide(dir_1[:, :1] * points[:, 1:] - dir_1[:, 1:] * points[:, :1], det)
    return bary_x, bary_y


def _get_normal_direction(direction: torch.Tensor, device) -> torch.Tensor:
    """Rotate a 2D direction 90° to obtain its unit outward normal.
    Mirrors ParallelogramBoundary._get_normal_direction() (parallelogram.py):
      swap x and y components, then negate the x-component.
    """
    normal = torch.index_select(direction, 1, torch.tensor([1, 0], device=device))
    normal[:, :1] *= -1
    return torch.divide(normal, torch.linalg.norm(normal, dim=1).reshape(-1, 1))


def _add_local_normal_vector(normals, bary_x, bary_y, normal_dir_1, normal_dir_2, i):
    """Add the normal contribution from the edge where bary == i (i=0 or i=1).
    Mirrors ParallelogramBoundary._add_local_normal_vector() (parallelogram.py):
      factor = 2*i - 1  (-1 for i=0, +1 for i=1)
      normals += normal_dir_1 * factor  where isclose(bary_y, i)
      normals += normal_dir_2 * factor  where isclose(bary_x, i)
    """
    y_close_i = torch.where(torch.isclose(bary_y, torch.tensor(i)), 2 * i - 1.0, 0.0)
    x_close_i = torch.where(torch.isclose(bary_x, torch.tensor(i)), 2 * i - 1.0, 0.0)
    normals += normal_dir_1 * y_close_i
    normals += normal_dir_2 * x_close_i


def _normals_square(x: torch.Tensor) -> torch.Tensor:
    """Outward unit normals on \\partial([0,W]x[0,H]).
    Mirrors ParallelogramBoundary.normal() (parallelogram.py).
    """
    origin, _, _, dir_1, dir_2 = _construct_parallelogram(x.shape[0], x.device)
    bary_x, bary_y = _solve_lgs(x - origin, dir_1, dir_2)
    normal_dir_1 =  _get_normal_direction(dir_1, x.device)
    normal_dir_2 = -_get_normal_direction(dir_2, x.device)
    n = torch.zeros_like(x)
    _add_local_normal_vector(n, bary_x, bary_y, normal_dir_1, normal_dir_2, 0.0)
    _add_local_normal_vector(n, bary_x, bary_y, normal_dir_1, normal_dir_2, 1.0)
    return torch.divide(n, torch.linalg.norm(n, dim=1).reshape(-1, 1))
'''

# ─────────────────────────────────────────────────────────────────────────────
# Template 1 — Poisson PINN (poisson-equation.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def _poisson_pinn_template(idx, variant, N):
    name = f"{idx:04d}_PoissonPINN_{variant}"
    N_bnd = N // 2
    code = f"""# Source  : torchphysics/examples/pinn/poisson-equation.ipynb
# Problem : PoissonPINN
# Variant : {variant}  (N_int={N}, N_bnd={N_bnd})
# PDE     : \\Delta u = 0   on  D = Circle(r=1) \\ Square([-0.25,0.25]^2)
# BC      : u = \\cos(x_1)\\cos(x_2)  on  \\partial D
# Network : FCN hidden=(20,20,20)
# Kernel  : one forward pass returns cat([\\Delta u(x_int), u(x_bnd)-f(x_bnd)])
# ─────────────────────────────────────────────────────────────────────────────
import math
import torch
import torch.nn as nn


def _f(x: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Dirichlet data  f(x) = \\cos(x_1) \\cos(x_2).
    https://en.wikipedia.org/wiki/Dirichlet_boundary_condition
    \"\"\"
    return torch.cos(x[:, :1]) * torch.cos(x[:, 1:])


{_LAPLACIAN_BLOCK}


class Model(nn.Module):
    \"\"\"
    Combined PINN residual for the Poisson equation.

    Mirrors the notebook: one network FCN(hidden=(20,20,20)) is shared by both the
    interior PDE condition (poisson_residual) and the boundary condition
    (boundary_residual), optimised together in a single Solver objective.

    Inputs  x_int (N_int, 2)  interior collocation points in D
            x_bnd (N_bnd, 2)  boundary collocation points on \\partial D
    Output  (N_int + N_bnd, 1)
            cat([\\Delta u(x_int),  u(x_bnd) - f(x_bnd)])
    \"\"\"

{_NET_BLOCK_20}
    def __init__(self):
        super().__init__()
        self.net = self._build_net(2, 1)

    def forward(self, x_int: torch.Tensor, x_bnd: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Args:
            x_int (torch.Tensor): interior points, shape (N_int, 2)
            x_bnd (torch.Tensor): boundary points, shape (N_bnd, 2)
        Returns:
            torch.Tensor: concatenated residuals of shape (N_int + N_bnd, 1)
        \"\"\"
        with torch.enable_grad():
            x_int = x_int.requires_grad_(True)
            u_int = self.net(x_int)
            lap   = _laplacian(u_int, x_int)
            u_bnd = self.net(x_bnd)
            bc    = u_bnd - _f(x_bnd)
        return torch.cat([lap, bc], dim=0)


def make_torchphysics_ref(model: Model):
    \"\"\"
    Return an nn.Module computing the combined residual via torchphysics.
    Interior uses tp.laplacian; boundary uses plain network evaluation.
    \"\"\"
    import torchphysics as tp

    class _Ref(nn.Module):
        def __init__(self, m):
            super().__init__()
            self._m = m

        def forward(self, x_int, x_bnd):
            with torch.enable_grad():
                x_int = x_int.requires_grad_(True)
                u_int = self._m.net(x_int)
                lap = tp.laplacian(u_int, x_int)
                u_bnd = self._m.net(x_bnd)
                bc = u_bnd - _f(x_bnd)
            return torch.cat([lap, bc], dim=0)

    return _Ref(model)


N_INT = {N}
N_BND = {N_bnd}


def get_inputs():
    torch.manual_seed(0)
    rng = torch.Generator()
    rng.manual_seed(0)

    # Interior: uniform inside Circle(r=1) \\ Square([-0.25,0.25]^2)
    pts = []
    while sum(p.shape[0] for p in pts) < N_INT:
        cands = torch.rand(N_INT * 8, 2, generator=rng) * 2.0 - 1.0
        in_circle = cands.norm(dim=-1) < 1.0
        in_square = (cands[:, 0].abs() < 0.25) & (cands[:, 1].abs() < 0.25)
        pts.append(cands[in_circle & ~in_square])
    x_int = torch.cat(pts)[:N_INT]

    # Boundary: outer circle (75 %) + inner square perimeter (25 %)
    n_outer = (N_BND * 3) // 4
    n_inner = N_BND - n_outer
    theta = torch.rand(n_outer, generator=rng) * 2.0 * math.pi
    x_outer = torch.stack([theta.cos(), theta.sin()], dim=-1)
    n_each = n_inner // 4
    t = torch.rand(n_each, generator=rng) * 0.5 - 0.25 # [-0.25, 0.25)
    edge = torch.full((n_each,), 0.25)
    x_inner = torch.cat([
        torch.stack([ t,  edge], dim=-1),
        torch.stack([ t, -edge], dim=-1),
        torch.stack([ edge,  t], dim=-1),
        torch.stack([-edge,  t], dim=-1),
    ])[:n_inner]
    x_bnd = torch.cat([x_outer, x_inner])

    return [x_int, x_bnd]


def get_init_inputs():
    return []
"""
    return name, code


# ─────────────────────────────────────────────────────────────────────────────
# Template 2 — Parametric Poisson PINN (poisson-with-input-params.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def _poisson_params_pinn_template(idx, variant, N):
    name = f"{idx:04d}_PoissonInputParamsPINN_{variant}"
    code = f"""# Source  : torchphysics/examples/pinn/poisson-with-input-params.ipynb
# Problem : PoissonInputParamsPINN
# Variant : {variant}  (N={N} per condition)
# PDE     : -D \\Delta u = g(x,F)  on  small_Rect = [0.6,0.8]^2
#           -D \\Delta u = 0        on  \\Omega \\ small_Rect
#            u = 0                  on  \\partial \\Omega
# Network : FCN(input=(x,D,F), hidden=(50,50,50,50,50))  — as in notebook
# Kernel  : one forward pass, cat([res_src, res_hom, res_bnd])
# ─────────────────────────────────────────────────────────────────────────────
import math
import torch
import torch.nn as nn

# Problem constants (from notebook)
D_LOW, D_UP = 0.1, 1.0
F_LOW, F_UP = 1.0, 5.0
_K  = math.pi / (2.0 * 0.1)   # wavenumber for source function g
_CX, _CY = 0.7, 0.7           # center of small_Rect


def _g(x: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Source function  g(x, F) = F \\cdot \\cos(k(x_1-c_x)) \\cdot \\cos(k(x_2-c_y)).
    \"\"\"
    return F * torch.cos(_K * (x[:, :1] - _CX)) * torch.cos(_K * (x[:, 1:] - _CY))


{_LAPLACIAN_BLOCK}


class Model(nn.Module):
    \"\"\"
    Combined parametric PINN residual for the Poisson equation.

    Mirrors the notebook: one FCN(hidden=(50,50,50,50,50)) with input (x,D,F)
    is shared by all three conditions (dirich_cond, pde_cond_1, pde_cond_2),
    optimised together in a single Solver objective.

    Inputs  x_src (N, 2)  source-region spatial points  [0.6,0.8]^2
            x_hom (N, 2)  homogeneous-region spatial points  \\Omega \\ small_Rect
            x_bnd (N, 2)  boundary spatial points  \\partial \\Omega
            D_vals (N, 1) diffusivity parameter  D \\in [0.1, 1.0]
            F_vals (N, 1) force magnitude        F \\in [1.0, 5.0]
    Output  (3N, 1)  cat([res_src, res_hom, res_bnd])
    \"\"\"

{_NET_BLOCK_50}
    def __init__(self):
        super().__init__()
        self.net = self._build_net(4, 1)   # input: (x_1, x_2, D, F)

    def forward(
        self,
        x_src:  torch.Tensor,
        x_hom:  torch.Tensor,
        x_bnd:  torch.Tensor,
        D_vals: torch.Tensor,
        F_vals: torch.Tensor,
    ) -> torch.Tensor:
        \"\"\"
        Args:
            x_src  (N, 2): source region collocation points
            x_hom  (N, 2): homogeneous region collocation points
            x_bnd  (N, 2): boundary collocation points
            D_vals (N, 1): diffusivity values
            F_vals (N, 1): force magnitude values
        Returns:
            torch.Tensor: concatenated residuals of shape (3N, 1)
        \"\"\"
        with torch.enable_grad():
            # x gets grad; D and F do not — \\Delta is w.r.t. x only
            x_src = x_src.requires_grad_(True)
            x_hom = x_hom.requires_grad_(True)

            # Source region: -D \\Delta u - g(x, F) = 0 
            inp_src = torch.cat([x_src, D_vals, F_vals], dim=-1)
            u_src   = self.net(inp_src)
            lap_src = _laplacian(u_src, x_src)
            res_src = -D_vals * lap_src - _g(x_src, F_vals)

            # Homogeneous region: -D \\Delta u = 0 
            inp_hom = torch.cat([x_hom, D_vals, F_vals], dim=-1)
            u_hom   = self.net(inp_hom)
            lap_hom = _laplacian(u_hom, x_hom)
            res_hom = -D_vals * lap_hom

            # Dirichlet Boundary Condition: u = 0
            inp_bnd = torch.cat([x_bnd, D_vals, F_vals], dim=-1)
            u_bnd   = self.net(inp_bnd)
            res_bnd = u_bnd

        return torch.cat([res_src, res_hom, res_bnd], dim=0)


def make_torchphysics_ref(model: Model):
    \"\"\"
    Return an nn.Module computing the combined residual via torchphysics.
    Interior conditions use tp.laplacian(u, x); BC uses plain evaluation.
    \"\"\"
    import torchphysics as tp

    class _Ref(nn.Module):
        def __init__(self, m):
            super().__init__()
            self._m = m

        def forward(self, x_src, x_hom, x_bnd, D_vals, F_vals):
            with torch.enable_grad():
                x_src = x_src.requires_grad_(True)
                x_hom = x_hom.requires_grad_(True)

                inp_src = torch.cat([x_src, D_vals, F_vals], dim=-1)
                u_src   = self._m.net(inp_src)
                lap_src = tp.laplacian(u_src, x_src)
                res_src = -D_vals * lap_src - _g(x_src, F_vals)

                inp_hom = torch.cat([x_hom, D_vals, F_vals], dim=-1)
                u_hom   = self._m.net(inp_hom)
                lap_hom = tp.laplacian(u_hom, x_hom)
                res_hom = -D_vals * lap_hom

                inp_bnd = torch.cat([x_bnd, D_vals, F_vals], dim=-1)
                u_bnd   = self._m.net(inp_bnd)
                res_bnd = u_bnd

            return torch.cat([res_src, res_hom, res_bnd], dim=0)

    return _Ref(model)


N = {N}


def get_inputs():
    torch.manual_seed(0)
    rng = torch.Generator()
    rng.manual_seed(0)

    # Source region: uniform in small_Rect = [0.6, 0.8]^2
    x_src = torch.rand(N, 2, generator=rng) * 0.2 + 0.6

    # Homogeneous region: \\Omega \\ small_Rect  (rejection sampling)
    # \\Omega = [0,1]^2 minus R2=[0.7,1]x[0,0.4] minus R3=[0,0.4]x[0.4,0.5] # [x1,x2]x[y1,y2]
    pts = []
    while sum(p.shape[0] for p in pts) < N:
        cands    = torch.rand(N * 6, 2, generator=rng)
        in_R2    = (cands[:, 0] >= 0.7) & (cands[:, 1] <= 0.4)
        in_R3    = (cands[:, 0] <= 0.4) & (cands[:, 1] >= 0.4) & (cands[:, 1] <= 0.5)
        in_small = ((cands[:, 0] >= 0.6) & (cands[:, 0] <= 0.8) &
                    (cands[:, 1] >= 0.6) & (cands[:, 1] <= 0.8))
        pts.append(cands[~in_R2 & ~in_R3 & ~in_small])
    x_hom = torch.cat(pts)[:N]

    # Boundary \\partial\\Omega for L-shaped domain
    # \\Omega = [0,1]^2 \\ R2=[0.7,1]x[0,0.4] \\ R3=[0,0.4]x[0.4,0.5]
    # 10 segments traced clockwise from (0,0), sampled proportional to arc length:
    _lens = [0.7, 0.4, 0.3, 0.6, 1.0, 0.5, 0.4, 0.1, 0.4, 0.4]   # total = 4.8
    _sn   = [max(1, round(N * l / 4.8)) for l in _lens]
    _sn[-1] += N - sum(_sn)
    _t = [torch.rand(n, generator=rng) for n in _sn]
    x_bnd = torch.cat([
        torch.stack([_t[0] * 0.7, torch.zeros(_sn[0])], dim=-1),               # y=0, x in [0.0,0.7]
        torch.stack([torch.full((_sn[1],), 0.7), _t[1] * 0.4], dim=-1),        # x=0.7, y in [0.0,0.4]  (R2 cut)
        torch.stack([0.7 + _t[2] * 0.3, torch.full((_sn[2],), 0.4)], dim=-1),  # y=0.4, x in [0.7,1.0]  (R2 cut)
        torch.stack([torch.ones(_sn[3]), 0.4 + _t[3] * 0.6], dim=-1),          # x=1,   y in [0.4,1.0]
        torch.stack([_t[4], torch.ones(_sn[4])], dim=-1),                      # y=1,   x in [0.0,1.0]
        torch.stack([torch.zeros(_sn[5]), 0.5 + _t[5] * 0.5], dim=-1),         # x=0,   y in [0.5,1.0]
        torch.stack([_t[6] * 0.4, torch.full((_sn[6],), 0.5)], dim=-1),        # y=0.5, x in [0.0,0.4]  (R3 cut)
        torch.stack([torch.full((_sn[7],), 0.4), 0.4 + _t[7] * 0.1], dim=-1),  # x=0.4, y in [0.4,0.5]  (R3 cut)
        torch.stack([_t[8] * 0.4, torch.full((_sn[8],), 0.4) ], dim=-1),       # y=0.4, x in [0.0,0.4]  (R3 cut)
        torch.stack([torch.zeros(_sn[9]), _t[9] * 0.4], dim=-1),               # x=0,   y in [0.0,0.4]
    ])

    # D \\in [D_LOW, D_UP],  F \\in [F_LOW, F_UP]
    D_vals = torch.rand(N, 1, generator=rng) * (D_UP - D_LOW) + D_LOW
    F_vals = torch.rand(N, 1, generator=rng) * (F_UP - F_LOW) + F_LOW

    return [x_src, x_hom, x_bnd, D_vals, F_vals]


def get_init_inputs():
    return []
"""
    return name, code


# ─────────────────────────────────────────────────────────────────────────────
# Template 3 — Signorini PINN (signorini-equation.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def _signorini_pinn_template(idx, variant, N):
    name  = f"{idx:04d}_SignoriniPINN_{variant}"
    # Scale from notebook counts (interior=6500, dir=500, neu=1400, con=5000)
    # N_con = N is the base; others derived proportionally so large=5000 is exact.
    N_con = N                      # x=10, y in [0,10], Length=10, density=500, num ponits = 5000  
    N_int = round(N * 6500 / 5000) # n_points=6500
    N_dir = round(N *  500 / 5000) # x=0, y in [0,10], Length=10, density=50, num ponits = 500 
    N_neu = round(N * 1400 / 5000) # y=0 and y=10, x in [0,10], Length=20, density=70, num ponits = 1400
    code = f"""# Source  : torchphysics/examples/pinn/signorini-equation.ipynb
# Problem : SignoriniPINN
# Variant : {variant}  (N_int={N_int}, N_dir={N_dir}, N_neu={N_neu}, N_con={N_con})
# PDE     : -\\Delta u = f/100   on  \\Omega = [0,10]^2,  f \\in [0.9, 2.0]
# BC      : u = 0 on  \\Gamma_D  (left, x_1=0)
#           n\\cdot\\nabla u = 0 on  \\Gamma_N  (top/bottom, x_2=0 or 10)
#           Signorini contact conditions on  \\Gamma_C  (right, x_1=10) :
#             u >= g(x),  -n\\cdot\\nabla u <= 0,  (n\\cdot\\nabla u)(u-g) = 0
#           g(x) = -0.1 - ((x_2 - 5)/5)^2  (parabolic obstacle)
# Network : NormalizationLayer + FCN(hidden=(50,50,50,50,50)), input dim=3 (x_1,x_2,f)
# Kernel  : one forward pass returns
#           cat([res_pde, res_dir, res_neu, res_obs, res_sign, res_comp])
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

# Problem constants (from notebook)
W, H = 10.0, 10.0   # domain [0,W] x [0,H]
F_LOW, F_HIGH = 0.9, 2.0     # force parameter range


def _g_obs(x: torch.Tensor) -> torch.Tensor:
    \"\"\"Obstacle  g(x) = -0.1 - ((x_2 - H/2) / 5)^2  on \\Gamma_C.\"\"\"
    return -0.1 - ((x[:, 1:] - H / 2.0) / 5.0) ** 2


{_LAPLACIAN_BLOCK}


{_NORMAL_DERIVATIVE_BLOCK}


{_NORMALS_SQUARE_BLOCK}


class _NormLayer(nn.Module):
    \"\"\"
    Mirrors tp.models.NormalizationLayer for input space X*F.
    Maps (x_1, x_2, f) from [0,W]x[0,H]x[F_LOW,F_HIGH] to [-1,1]^3
    via  out_d = (inp_d - center_d) / scale_d.
    from torchphysics/src/torchphysics/models/normalization_layer.py
    \"\"\"
    def __init__(self):
        super().__init__()
        centers = torch.tensor([W / 2.0, H / 2.0, (F_LOW + F_HIGH) / 2.0])
        scales  = torch.tensor([W / 2.0, H / 2.0, (F_HIGH - F_LOW) / 2.0])
        self.register_buffer('centers', centers)
        self.register_buffer('scales',  scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.centers) / self.scales


class Model(nn.Module):
    \"\"\"
    Combined Signorini PINN residual.

    Mirrors the notebook: one network (NormalizationLayer + FCN(hidden=(50,50,50,50,50)))
    with input (x,f) is shared by all six PINNConditions and optimised in one Solver.

    Inputs  x_int (N_int, 2)  interior collocation points in \\Omega
            f_int (N_int, 1)  force parameter for interior points
            x_dir (N_dir, 2)  Dirichlet boundary points  (x_1 = 0)
            f_dir (N_dir, 1)  force parameter for Dirichlet points
            x_neu (N_neu, 2)  Neumann boundary points  (x_2 = 0 or x_2 = H)
            f_neu (N_neu, 1)  force parameter for Neumann points
            x_con (N_con, 2)  contact boundary points  (x_1 = W)
            f_con (N_con, 1)  force parameter for contact points
    Output  (N_int + N_dir + N_neu + 3*N_con, 1)
            cat([res_pde, res_dir, res_neu, res_obs, res_sign, res_comp])
    \"\"\"

{_NET_BLOCK_50}
    def __init__(self):
        super().__init__()
        # NormalizationLayer prepended to the FCN, mirrors tp.models.Sequential(norm, fcn)
        self.net = nn.Sequential(_NormLayer(), self._build_net(3, 1))

    def forward(
        self,
        x_int: torch.Tensor, f_int: torch.Tensor,
        x_dir: torch.Tensor, f_dir: torch.Tensor,
        x_neu: torch.Tensor, f_neu: torch.Tensor,
        x_con: torch.Tensor, f_con: torch.Tensor,
    ) -> torch.Tensor:
        \"\"\"
        Args:
            x_int (N_int, 2): interior collocation points
            f_int (N_int, 1): force parameter for interior points
            x_dir (N_dir, 2): Dirichlet boundary points (x_1=0)
            f_dir (N_dir, 1): force parameter for Dirichlet points
            x_neu (N_neu, 2): Neumann boundary points (x_2=0 or x_2=H)
            f_neu (N_neu, 1): force parameter for Neumann points
            x_con (N_con, 2): contact boundary points (x_1=W)
            f_con (N_con, 1): force parameter for contact points
        Returns:
            torch.Tensor: concatenated residuals, shape (N_int+N_dir+N_neu+3*N_con, 1)
        \"\"\"
        with torch.enable_grad():
            x_int = x_int.requires_grad_(True)
            x_neu = x_neu.requires_grad_(True)
            x_con = x_con.requires_grad_(True)

            # Interior PDE: -\\Delta u = f/100 
            u_int = self.net(torch.cat([x_int, f_int], dim=-1))
            lap = _laplacian(u_int, x_int)
            res_pde = -lap - f_int / 100.0

            # Dirichlet boundary condition: u = 0 on \\Gamma_D (x_1 = 0) 
            u_dir   = self.net(torch.cat([x_dir, f_dir], dim=-1))
            res_dir = u_dir

            # Neumann boundary condition: n\\cdot\\nabla u = 0 on \\Gamma_N 
            u_neu   = self.net(torch.cat([x_neu, f_neu], dim=-1))
            res_neu = _normal_derivative(u_neu, _normals_square(x_neu), x_neu)

            # Contact conditions on \\Gamma_C (x_1 = W) 
            # outward normal at x_1=W is (1, 0), so n\\cdot\\nabla u = \\partial u/\\partial x_1
            u_con    = self.net(torch.cat([x_con, f_con], dim=-1))
            g_con    = _g_obs(x_con)
            grad_con = torch.autograd.grad(u_con.sum(), x_con, create_graph=True)[0]
            du_dn    = grad_con[:, :1]

            # obstacle:        clamp(u - g, max=0)  * sqrt(10)  — weight=10 in notebook
            res_obs  = torch.clamp(u_con - g_con, max=0.0) * 10 ** 0.5

            # sign:            clamp(du/dn, max=0)              — weight=1 (default)
            res_sign = torch.clamp(du_dn, max=0.0)

            # complementarity: du/dn * (u - g)     * sqrt(10)  — weight=10 in notebook
            res_comp = du_dn * (u_con - g_con) * 10 ** 0.5

        return torch.cat([res_pde, res_dir, res_neu, res_obs, res_sign, res_comp], dim=0)


def make_torchphysics_ref(model: Model):
    \"\"\"
    Return an nn.Module computing the combined residual via torchphysics.
    Uses tp.utils.laplacian and tp.utils.grad.
    \"\"\"
    import torchphysics as tp

    class _Ref(nn.Module):
        def __init__(self, m):
            super().__init__()
            self._m = m

        def forward(self, x_int, f_int, x_dir, f_dir, x_neu, f_neu, x_con, f_con):
            with torch.enable_grad():
                x_int = x_int.requires_grad_(True)
                x_neu = x_neu.requires_grad_(True)
                x_con = x_con.requires_grad_(True)

                u_int   = self._m.net(torch.cat([x_int, f_int], dim=-1))
                lap     = tp.utils.laplacian(u_int, x_int)
                res_pde = -lap - f_int / 100.0

                u_dir   = self._m.net(torch.cat([x_dir, f_dir], dim=-1))
                res_dir = u_dir

                u_neu   = self._m.net(torch.cat([x_neu, f_neu], dim=-1))
                res_neu = tp.utils.normal_derivative(u_neu, _normals_square(x_neu), x_neu)

                u_con    = self._m.net(torch.cat([x_con, f_con], dim=-1))
                g_con    = _g_obs(x_con)
                grad_con = tp.utils.grad(u_con, x_con)
                du_dn    = grad_con[:, :1]

                res_obs  = torch.clamp(u_con - g_con, max=0.0) * 10 ** 0.5
                res_sign = torch.clamp(du_dn, max=0.0)
                res_comp = du_dn * (u_con - g_con) * 10 ** 0.5

            return torch.cat([res_pde, res_dir, res_neu, res_obs, res_sign, res_comp], dim=0)

    return _Ref(model)


N_INT = {N_int}
N_DIR = {N_dir}
N_NEU = {N_neu}
N_CON = {N_con}


def get_inputs():
    torch.manual_seed(0)
    rng = torch.Generator()
    rng.manual_seed(0)

    # Interior: uniform in [0, W] x [0, H]
    x_int = torch.rand(N_INT, 2, generator=rng) * torch.tensor([W, H])
    f_int = torch.rand(N_INT, 1, generator=rng) * (F_HIGH - F_LOW) + F_LOW

    # Dirichlet boundary: x_1 = 0,  x_2 in [0, H]
    x_dir = torch.stack([
        torch.zeros(N_DIR),
        torch.rand(N_DIR, generator=rng) * H,
    ], dim=-1)
    f_dir = torch.rand(N_DIR, 1, generator=rng) * (F_HIGH - F_LOW) + F_LOW

    # Neumann boundary: x_2 = 0 (bottom) or x_2 = H (top),  x_1 in [0, W]
    n_bot = N_NEU // 2
    n_top = N_NEU - n_bot
    x_neu = torch.cat([
        torch.stack([torch.rand(n_bot, generator=rng) * W, torch.zeros(n_bot)],          dim=-1),
        torch.stack([torch.rand(n_top, generator=rng) * W, torch.full((n_top,), H)],     dim=-1),
    ])
    f_neu = torch.rand(N_NEU, 1, generator=rng) * (F_HIGH - F_LOW) + F_LOW

    # Contact boundary: x_1 = W,  x_2 in [0, H]
    x_con = torch.stack([
        torch.full((N_CON,), W),
        torch.rand(N_CON, generator=rng) * H,
    ], dim=-1)
    f_con = torch.rand(N_CON, 1, generator=rng) * (F_HIGH - F_LOW) + F_LOW

    return [x_int, f_int, x_dir, f_dir, x_neu, f_neu, x_con, f_con]


def get_init_inputs():
    return []
"""
    return name, code


# ─────────────────────────────────────────────────────────────────────────────
# Reference mapping
# ─────────────────────────────────────────────────────────────────────────────

_NORMLAYER = (
    "torchphysics/src/torchphysics/models/normalization_layer.py"
    " (NormalizationLayer)"
)

_TP_REFS = {
    _poisson_pinn_template:        [f"{_DIFFOPS}:11 (laplacian)"],
    _poisson_params_pinn_template:  [f"{_DIFFOPS}:11 (laplacian)"],
    _signorini_pinn_template:       [f"{_DIFFOPS}:11 (laplacian)", _NORMLAYER],
}

# ─────────────────────────────────────────────────────────────────────────────
# Output directories
# ─────────────────────────────────────────────────────────────────────────────

_BASE = Path(__file__).resolve().parent.parent / "tasks" / "sci_bench_torchphysics" / "level2"

POISSON_DIR   = _BASE / "poisson_pinn"
PARAMS_DIR    = _BASE / "poisson_input_params_pinn"
SIGNORINI_DIR = _BASE / "signorini_pinn"

# ─────────────────────────────────────────────────────────────────────────────
# Task list — (factory, out_dir, variant, N)
# Each out_dir gets its own 0001/0002/0003 numbering.
# ─────────────────────────────────────────────────────────────────────────────

TASKS = [
    (_poisson_pinn_template,        POISSON_DIR,   "small",  128),
    (_poisson_pinn_template,        POISSON_DIR,   "medium", 512),
    (_poisson_pinn_template,        POISSON_DIR,   "large",  1024),
    (_poisson_params_pinn_template, PARAMS_DIR,    "small",  128),
    (_poisson_params_pinn_template, PARAMS_DIR,    "medium", 512),
    (_poisson_params_pinn_template, PARAMS_DIR,    "large",  1024),
    (_signorini_pinn_template,      SIGNORINI_DIR, "small",  1250),
    (_signorini_pinn_template,      SIGNORINI_DIR, "medium", 2500),
    (_signorini_pinn_template,      SIGNORINI_DIR, "large",  5000),
]

# ─────────────────────────────────────────────────────────────────────────────
# Generation entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate():
    # Prepare output directories and clear old files
    for out_dir in (POISSON_DIR, PARAMS_DIR, SIGNORINI_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("[0-9]*.py"):
            old.unlink()

    # Per-directory index counter
    dir_idx: dict[Path, int] = {}

    for factory, out_dir, variant, N in TASKS:
        idx = dir_idx.get(out_dir, 0) + 1
        dir_idx[out_dir] = idx

        filename, code = factory(idx, variant, N)

        refs = _TP_REFS.get(factory, [])
        ref_lines = "\n".join(f"# torchphysics_ref : {r}" for r in refs)
        ref_block = _BUILD_NET_REF + ("\n" + ref_lines if refs else "")
        code = ref_block + "\n" + code

        path = out_dir / f"{filename}.py"
        path.write_text(code)
        print(f"  wrote {path.relative_to(_BASE.parent.parent)}")

    total = len(TASKS)
    print(f"\nGenerated {total} tasks across {len(dir_idx)} directories")


if __name__ == "__main__":
    generate()
