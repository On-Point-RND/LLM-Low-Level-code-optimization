#!/usr/bin/env python3
"""
generate_scientifickernelbench_torchphysics_level1_functional.py
================================================================
Generates Level-1 benchmark tasks rewritten to use torch.func
(vmap/jacrev/hessian + functional_call) instead of explicit
torch.autograd.grad loops.

The goal is to preserve the same mathematical operators and
numerical outputs up to normal floating-point tolerance, while
making the computation graph more tensorized / compiler-friendly.

Tasks are written to:
  ScientificKernelBenchFunctional/level1/

Run:
    python scripts/generate_scientifickernelbench_torchphysics_level1_functional.py
"""

from pathlib import Path

_BUILD_NET_REF = (
    "# _build_net     : adapted from tp.models.FCN\n"
    "#   torchphysics/src/torchphysics/models/fcn.py:9  (_construct_FC_layers)\n"
    "#   torchphysics/src/torchphysics/models/fcn.py:61 (FCN.__init__)\n"
    "#   differences: accepts plain int dims instead of Space objects;\n"
    "#   returns nn.Sequential (not Points-wrapped) for functional torch.func use"
)

_DIFFOPS = (
    "torchphysics/src/torchphysics/utils/differentialoperators/"
    "differentialoperators.py"
)

_IMPORTS_BLOCK = '''\
import torch
import torch.nn as nn
from torch.func import functional_call, vmap, jacrev, hessian
'''

_NET_BLOCK = """\
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
"""

_FUNC_HELPERS_BLOCK = '''\
def _make_functional_net(net: nn.Module):
    params = dict(net.named_parameters())
    buffers = dict(net.named_buffers())

    def net_single(x_single: torch.Tensor) -> torch.Tensor:
        out = functional_call(net, (params, buffers), (x_single.unsqueeze(0),))
        return out.squeeze(0)

    return net_single


def _scalarize_last_dim(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 1 and y.shape[0] == 1:
        return y[0]
    return y.squeeze()


def _trace_last_two(mat: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(mat, dim1=-2, dim2=-1).sum(dim=-1)


def _symmetrize_last_two(mat: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mat + mat.transpose(-2, -1))
'''

def _coord_desc(d):
    return ",".join("xyz"[:d]) if d <= 3 else ",".join(f"x{i+1}" for i in range(d))


def _laplacian_template(d):
    xd = _coord_desc(d)

    def factory(idx, variant, N, hidden):
        name = f"{idx:04d}_L1Laplacian_{d}d_{variant}"
        code = f"""# Problem : L1Laplacian_{d}d
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: laplacian (L11)  —  \\Delta u({xd}) = 0
# ─────────────────────────────────────────────────────────────────────────────
{_IMPORTS_BLOCK}

{_FUNC_HELPERS_BLOCK}

class Model(nn.Module):
    \"\"\"
    Laplacian of a network \\Delta u with respect to the given variable ({d}D),
    implemented with torch.func for compiler-friendly higher-order derivatives.
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net({d}, {hidden}, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net_single = _make_functional_net(self.net)

        def scalar_field(x_single: torch.Tensor) -> torch.Tensor:
            return _scalarize_last_dim(net_single(x_single))

        H = vmap(hessian(scalar_field))(x)
        lap = _trace_last_two(H).unsqueeze(-1)
        return lap


def make_torchphysics_ref(model: Model):
    \"\"\"Return an nn.Module that computes \\Delta u via torchphysics ({d}D).\"\"\"
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


N      = {N}
hidden = {hidden}


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, {d})
    return [x]


def get_init_inputs():
    return []
"""
        return name, code

    return factory


def _grad_template(d):
    xd = _coord_desc(d)

    def factory(idx, variant, N, hidden):
        name = f"{idx:04d}_L1Grad_{d}d_{variant}"
        code = f"""# Problem : L1Grad_{d}d
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: grad (L47)  —  \\nabla u({xd})
# ─────────────────────────────────────────────────────────────────────────────
{_IMPORTS_BLOCK}

{_FUNC_HELPERS_BLOCK}

class Model(nn.Module):
    \"\"\"
    Gradient of a scalar network output \\nabla u ({d}D), implemented with torch.func.
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net({d}, {hidden}, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net_single = _make_functional_net(self.net)

        def scalar_field(x_single: torch.Tensor) -> torch.Tensor:
            return _scalarize_last_dim(net_single(x_single))

        return vmap(jacrev(scalar_field))(x)


def make_torchphysics_ref(model: Model):
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


N      = {N}
hidden = {hidden}


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, {d})
    return [x]


def get_init_inputs():
    return []
"""
        return name, code

    return factory


def _normal_deriv_template(d):
    def factory(idx, variant, N, hidden):
        name = f"{idx:04d}_L1NormalDeriv_{d}d_{variant}"
        code = f"""# Problem : L1NormalDeriv_{d}d
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: normal_derivative (L111)  —  \\partial u / \\partial n = \\nabla u \\cdot n
# ─────────────────────────────────────────────────────────────────────────────
{_IMPORTS_BLOCK}
import torch.nn.functional as F

{_FUNC_HELPERS_BLOCK}

class Model(nn.Module):
    \"\"\"
    Normal derivative \\partial u / \\partial n via torch.func ({d}D).
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net({d}, {hidden}, 1)

    def forward(self, x: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        net_single = _make_functional_net(self.net)

        def scalar_field(x_single: torch.Tensor) -> torch.Tensor:
            return _scalarize_last_dim(net_single(x_single))

        gradient = vmap(jacrev(scalar_field))(x)
        return (gradient * normals).sum(dim=-1, keepdim=True)


def make_torchphysics_ref(model: Model):
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


N      = {N}
hidden = {hidden}


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, {d})
    normals = F.normalize(torch.randn(N, {d}), dim=-1)
    return [x, normals]


def get_init_inputs():
    return []
"""
        return name, code

    return factory


def _div_template(d):
    xd = _coord_desc(d)

    def factory(idx, variant, N, hidden):
        name = f"{idx:04d}_L1Div_{d}d_{variant}"
        code = f"""# Problem : L1Div_{d}d
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: div (L137)  —  \\nabla \\cdot u({xd})
# ─────────────────────────────────────────────────────────────────────────────
{_IMPORTS_BLOCK}

{_FUNC_HELPERS_BLOCK}

class Model(nn.Module):
    \"\"\"
    Divergence of a vector field via torch.func ({d}D).
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net({d}, {hidden}, {d})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net_single = _make_functional_net(self.net)
        J = vmap(jacrev(net_single))(x)
        return _trace_last_two(J).unsqueeze(-1)


def make_torchphysics_ref(model: Model):
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


N      = {N}
hidden = {hidden}


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, {d})
    return [x]


def get_init_inputs():
    return []
"""
        return name, code

    return factory


def _jac_template(d):
    xd = _coord_desc(d)

    def factory(idx, variant, N, hidden):
        name = f"{idx:04d}_L1Jac_{d}d_{variant}"
        code = f"""# Problem : L1Jac_{d}d
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: jac (L233)  —  J(u)({xd}), u:(N,{d})
# ─────────────────────────────────────────────────────────────────────────────
{_IMPORTS_BLOCK}

{_FUNC_HELPERS_BLOCK}

class Model(nn.Module):
    \"\"\"
    Jacobian J(u) of a vector-valued network output via torch.func ({d}D).
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net({d}, {hidden}, {d})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net_single = _make_functional_net(self.net)
        return vmap(jacrev(net_single))(x)


def make_torchphysics_ref(model: Model):
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


N      = {N}
hidden = {hidden}


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, {d})
    return [x]


def get_init_inputs():
    return []
"""
        return name, code

    return factory


def _rot3d_template(idx, variant, N, hidden):
    name = f"{idx:04d}_L1Rot_3d_{variant}"
    code = f"""# Problem : L1Rot_3d
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: rot (L261)  —  \\nabla \\times u(x,y,z), u:(N,3)
# ─────────────────────────────────────────────────────────────────────────────
{_IMPORTS_BLOCK}

{_FUNC_HELPERS_BLOCK}

class Model(nn.Module):
    \"\"\"
    Rotation / curl \\nabla \\times u of a 3D vector field via torch.func.
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net(3, {hidden}, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net_single = _make_functional_net(self.net)
        jacobian = vmap(jacrev(net_single))(x)
        rotation = x.new_zeros((x.shape[0], 3))
        rotation[:, 0] = jacobian[:, 2, 1] - jacobian[:, 1, 2]
        rotation[:, 1] = jacobian[:, 0, 2] - jacobian[:, 2, 0]
        rotation[:, 2] = jacobian[:, 1, 0] - jacobian[:, 0, 1]
        return rotation


def make_torchphysics_ref(model: Model):
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


N      = {N}
hidden = {hidden}


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, 3)
    return [x]


def get_init_inputs():
    return []
"""
    return name, code


def _partial_template(d):
    def factory(idx, variant, N, hidden):
        name = f"{idx:04d}_L1Partial_{d}dt_{variant}"
        code = f"""# Problem : L1Partial_{d}dt
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: partial (L294)  —  \\partial u/\\partial t, spatial dim={d}
# ─────────────────────────────────────────────────────────────────────────────
{_IMPORTS_BLOCK}

{_FUNC_HELPERS_BLOCK}

class Model(nn.Module):
    \"\"\"
    Partial derivative \\partial u/\\partial t via torch.func ({d}D+t).
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net({d + 1}, {hidden}, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        net_single = _make_functional_net(self.net)

        def scalar_field(xt_single: torch.Tensor) -> torch.Tensor:
            return _scalarize_last_dim(net_single(xt_single))

        xt = torch.cat([x, t], dim=-1)
        grad_xt = vmap(jacrev(scalar_field))(xt)
        return grad_xt[:, -1:].contiguous()


def make_torchphysics_ref(model: Model):
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


N      = {N}
hidden = {hidden}


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, {d})
    t = torch.rand(N, 1)
    return [x, t]


def get_init_inputs():
    return []
"""
        return name, code

    return factory


def _convective_template(d):
    def factory(idx, variant, N, hidden):
        name = f"{idx:04d}_L1Convective_{d}d_{variant}"
        code = f"""# Problem : L1Convective_{d}d
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: convective (L320)  —  (v \\cdot \\nabla)u, self-advection in {d}D
# ─────────────────────────────────────────────────────────────────────────────
{_IMPORTS_BLOCK}

{_FUNC_HELPERS_BLOCK}

class Model(nn.Module):
    \"\"\"
    Convective term (v \\cdot \\nabla)u via torch.func ({d}D).
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net({d}, {hidden}, {d})

    def forward(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        net_single = _make_functional_net(self.net)
        jac_x = vmap(jacrev(net_single))(x)
        return torch.bmm(jac_x, v.unsqueeze(-1)).squeeze(-1)


def make_torchphysics_ref(model: Model):
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


N      = {N}
hidden = {hidden}


def get_inputs():
    torch.manual_seed(0)
    v = torch.randn(N, {d})
    x = torch.randn(N, {d})
    return [v, x]


def get_init_inputs():
    return []
"""
        return name, code

    return factory


def _sym_grad_template(d):
    def factory(idx, variant, N, hidden):
        name = f"{idx:04d}_L1SymGrad_{d}d_{variant}"
        code = f"""# Problem : L1SymGrad_{d}d
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: sym_grad (L345)  —  \\varepsilon(u) = 1/2(\\nabla u + \\nabla u^T) in {d}D
# ─────────────────────────────────────────────────────────────────────────────
{_IMPORTS_BLOCK}

{_FUNC_HELPERS_BLOCK}

class Model(nn.Module):
    \"\"\"
    Symmetric gradient 0.5(\\nabla u + \\nabla u^T) via torch.func ({d}D).
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net({d}, {hidden}, {d})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net_single = _make_functional_net(self.net)
        jac_matrix = vmap(jacrev(net_single))(x)
        return _symmetrize_last_two(jac_matrix)


def make_torchphysics_ref(model: Model):
    import torchphysics as tp

    class _Ref(nn.Module):
        def __init__(self, m):
            super().__init__()
            self._m = m

        def forward(self, x):
            with torch.enable_grad():
                x = x.requires_grad_(True)
                u = self._m.net(x)
                return tp.sym_grad(u, x)

    return _Ref(model)


N      = {N}
hidden = {hidden}


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, {d})
    return [x]


def get_init_inputs():
    return []
"""
        return name, code

    return factory


def _matrix_div_template(d):
    out_dim = d * d

    def factory(idx, variant, N, hidden):
        name = f"{idx:04d}_L1MatrixDiv_{d}d_{variant}"
        code = f"""# Problem : L1MatrixDiv_{d}d
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: matrix_div (L365)  —  \\nabla \\cdot \\sigma, \\sigma:(N,{d},{d}) in {d}D
# ─────────────────────────────────────────────────────────────────────────────
{_IMPORTS_BLOCK}

{_FUNC_HELPERS_BLOCK}

class Model(nn.Module):
    \"\"\"
    Matrix divergence \\nabla \\cdot \\sigma via torch.func ({d}D).

    Net outputs sigma_flat:(N,{out_dim}) reshaped to sigma:(N,{d},{d}).
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net({d}, {hidden}, {out_dim})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net_single = _make_functional_net(self.net)

        def sigma_single(x_single: torch.Tensor) -> torch.Tensor:
            return net_single(x_single).view({d}, {d})

        J_sigma = vmap(jacrev(sigma_single))(x)
        return torch.diagonal(J_sigma, dim1=-2, dim2=-1).sum(dim=-1)


def make_torchphysics_ref(model: Model):
    import torchphysics as tp

    class _Ref(nn.Module):
        def __init__(self, m):
            super().__init__()
            self._m = m

        def forward(self, x):
            with torch.enable_grad():
                x = x.requires_grad_(True)
                sigma = self._m.net(x).view(-1, {d}, {d})
                return tp.matrix_div(sigma, x)

    return _Ref(model)


N      = {N}
hidden = {hidden}


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, {d})
    return [x]


def get_init_inputs():
    return []
"""
        return name, code

    return factory


_laplacian2d_template = _laplacian_template(2)
_laplacian3d_template = _laplacian_template(3)
_laplacian4d_template = _laplacian_template(4)

_grad2d_template      = _grad_template(2)
_grad3d_template      = _grad_template(3)
_grad4d_template      = _grad_template(4)

_normal_deriv2d_template = _normal_deriv_template(2)
_normal_deriv3d_template = _normal_deriv_template(3)
_normal_deriv4d_template = _normal_deriv_template(4)

_div2d_template = _div_template(2)
_div3d_template = _div_template(3)
_div4d_template = _div_template(4)

_jac2d_template = _jac_template(2)
_jac3d_template = _jac_template(3)
_jac4d_template = _jac_template(4)

_partial1dt_template = _partial_template(1)
_partial2dt_template = _partial_template(2)
_partial3dt_template = _partial_template(3)

_convective2d_template = _convective_template(2)
_convective3d_template = _convective_template(3)
_convective4d_template = _convective_template(4)

_sym_grad2d_template = _sym_grad_template(2)
_sym_grad3d_template = _sym_grad_template(3)
_sym_grad4d_template = _sym_grad_template(4)

_matrix_div2d_template = _matrix_div_template(2)
_matrix_div3d_template = _matrix_div_template(3)
_matrix_div4d_template = _matrix_div_template(4)


_TP_L1_REFS = {
    _laplacian2d_template:    [f"{_DIFFOPS}:11 (laplacian)"],
    _laplacian3d_template:    [f"{_DIFFOPS}:11 (laplacian)"],
    _laplacian4d_template:    [f"{_DIFFOPS}:11 (laplacian)"],
    _grad2d_template:         [f"{_DIFFOPS}:47 (grad)"],
    _grad3d_template:         [f"{_DIFFOPS}:47 (grad)"],
    _grad4d_template:         [f"{_DIFFOPS}:47 (grad)"],
    _normal_deriv2d_template: [f"{_DIFFOPS}:111 (normal_derivative)"],
    _normal_deriv3d_template: [f"{_DIFFOPS}:111 (normal_derivative)"],
    _normal_deriv4d_template: [f"{_DIFFOPS}:111 (normal_derivative)"],
    _div2d_template:          [f"{_DIFFOPS}:137 (div)"],
    _div3d_template:          [f"{_DIFFOPS}:137 (div)"],
    _div4d_template:          [f"{_DIFFOPS}:137 (div)"],
    _jac2d_template:          [f"{_DIFFOPS}:233 (jac)"],
    _jac3d_template:          [f"{_DIFFOPS}:233 (jac)"],
    _jac4d_template:          [f"{_DIFFOPS}:233 (jac)"],
    _rot3d_template:          [f"{_DIFFOPS}:261 (rot)"],
    _partial1dt_template:     [f"{_DIFFOPS}:294 (partial)"],
    _partial2dt_template:     [f"{_DIFFOPS}:294 (partial)"],
    _partial3dt_template:     [f"{_DIFFOPS}:294 (partial)"],
    _convective2d_template:   [f"{_DIFFOPS}:320 (convective)"],
    _convective3d_template:   [f"{_DIFFOPS}:320 (convective)"],
    _convective4d_template:   [f"{_DIFFOPS}:320 (convective)"],
    _sym_grad2d_template:     [f"{_DIFFOPS}:345 (sym_grad)"],
    _sym_grad3d_template:     [f"{_DIFFOPS}:345 (sym_grad)"],
    _sym_grad4d_template:     [f"{_DIFFOPS}:345 (sym_grad)"],
    _matrix_div2d_template:   [f"{_DIFFOPS}:365 (matrix_div)"],
    _matrix_div3d_template:   [f"{_DIFFOPS}:365 (matrix_div)"],
    _matrix_div4d_template:   [f"{_DIFFOPS}:365 (matrix_div)"],
}

TASKS = [
    (_laplacian2d_template, "small",  128,  32), (_laplacian2d_template, "medium", 512,  64), (_laplacian2d_template, "large",  1024, 128),
    (_laplacian3d_template, "small",  128,  32), (_laplacian3d_template, "medium", 512,  64), (_laplacian3d_template, "large",  1024, 128),
    (_laplacian4d_template, "small",  128,  32), (_laplacian4d_template, "medium", 512,  64), (_laplacian4d_template, "large",  1024, 128),
    (_grad2d_template, "small",  128,  32), (_grad2d_template, "medium", 512,  64), (_grad2d_template, "large",  1024, 128),
    (_grad3d_template, "small",  128,  32), (_grad3d_template, "medium", 512,  64), (_grad3d_template, "large",  1024, 128),
    (_grad4d_template, "small",  128,  32), (_grad4d_template, "medium", 512,  64), (_grad4d_template, "large",  1024, 128),
    (_normal_deriv2d_template, "small",  128,  32), (_normal_deriv2d_template, "medium", 512,  64), (_normal_deriv2d_template, "large",  1024, 128),
    (_normal_deriv3d_template, "small",  128,  32), (_normal_deriv3d_template, "medium", 512,  64), (_normal_deriv3d_template, "large",  1024, 128),
    (_normal_deriv4d_template, "small",  128,  32), (_normal_deriv4d_template, "medium", 512,  64), (_normal_deriv4d_template, "large",  1024, 128),
    (_div2d_template, "small",  128,  32), (_div2d_template, "medium", 512,  64), (_div2d_template, "large",  1024, 128),
    (_div3d_template, "small",  128,  32), (_div3d_template, "medium", 512,  64), (_div3d_template, "large",  1024, 128),
    (_div4d_template, "small",  128,  32), (_div4d_template, "medium", 512,  64), (_div4d_template, "large",  1024, 128),
    (_jac2d_template, "small",  128,  32), (_jac2d_template, "medium", 512,  64), (_jac2d_template, "large",  1024, 128),
    (_jac3d_template, "small",  128,  32), (_jac3d_template, "medium", 512,  64), (_jac3d_template, "large",  1024, 128),
    (_jac4d_template, "small",  128,  32), (_jac4d_template, "medium", 512,  64), (_jac4d_template, "large",  1024, 128),
    (_rot3d_template, "small",  128,  32), (_rot3d_template, "medium", 512,  64), (_rot3d_template, "large",  1024, 128),
    (_partial1dt_template, "small",  128,  32), (_partial1dt_template, "medium", 512,  64), (_partial1dt_template, "large",  1024, 128),
    (_partial2dt_template, "small",  128,  32), (_partial2dt_template, "medium", 512,  64), (_partial2dt_template, "large",  1024, 128),
    (_partial3dt_template, "small",  128,  32), (_partial3dt_template, "medium", 512,  64), (_partial3dt_template, "large",  1024, 128),
    (_convective2d_template, "small",  128,  32), (_convective2d_template, "medium", 512,  64), (_convective2d_template, "large",  1024, 128),
    (_convective3d_template, "small",  128,  32), (_convective3d_template, "medium", 512,  64), (_convective3d_template, "large",  1024, 128),
    (_convective4d_template, "small",  128,  32), (_convective4d_template, "medium", 512,  64), (_convective4d_template, "large",  1024, 128),
    (_sym_grad2d_template, "small",  128,  32), (_sym_grad2d_template, "medium", 512,  64), (_sym_grad2d_template, "large",  1024, 128),
    (_sym_grad3d_template, "small",  128,  32), (_sym_grad3d_template, "medium", 512,  64), (_sym_grad3d_template, "large",  1024, 128),
    (_sym_grad4d_template, "small",  128,  32), (_sym_grad4d_template, "medium", 512,  64), (_sym_grad4d_template, "large",  1024, 128),
    (_matrix_div2d_template, "small",  128,  32), (_matrix_div2d_template, "medium", 512,  64), (_matrix_div2d_template, "large",  1024, 128),
    (_matrix_div3d_template, "small",  128,  32), (_matrix_div3d_template, "medium", 512,  64), (_matrix_div3d_template, "large",  1024, 128),
    (_matrix_div4d_template, "small",  128,  32), (_matrix_div4d_template, "medium", 512,  64), (_matrix_div4d_template, "large",  1024, 128),
]

OUT_DIR = Path(__file__).resolve().parent.parent / "ScientificKernelBenchFunctional" / "level1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate():
    for old in OUT_DIR.glob("[0-9]*.py"):
        old.unlink()
    for idx, (factory, variant, N, hidden) in enumerate(TASKS, start=1):
        filename, code = factory(idx, variant, N, hidden)
        refs = _TP_L1_REFS.get(factory, [])
        ref_block = _BUILD_NET_REF + "\n" + "\n".join(f"# torchphysics_ref : {r}" for r in refs)
        code = ref_block + "\n" + code
        path = OUT_DIR / f"{filename}.py"
        path.write_text(code)
        print(f"  wrote {path.name}")
    print(f"\nGenerated {len(TASKS)} tasks in {OUT_DIR}")


if __name__ == "__main__":
    generate()
