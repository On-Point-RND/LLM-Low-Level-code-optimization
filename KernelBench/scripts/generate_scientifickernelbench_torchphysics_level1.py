#!/usr/bin/env python3
"""
generate_sci_bench_torchphysics_level1.py
==========================================
Generates Level-1 benchmark tasks that each isolate **one** torchphysics
differential operator primitive.

Tasks are written to:
  KernelBench/ScientificKernelBench/level1/

Run:
    python scripts/generate_scientifickernelbench_torchphysics_level1.py
"""

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

_GRAD_BLOCK = """\
def _grad(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    grads = []
    for vari in [x]:
        new_grad = torch.autograd.grad(u.sum(), vari, create_graph=True)[0]
        grads.append(new_grad)
    return torch.column_stack(grads)
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
    return laplacian
'''

_DIV_BLOCK = '''
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
'''

_JAC_BLOCK = '''
def _jac(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    Du_rows = []
    for i in range(u.shape[1]):
        Du_i = []
        for vari in [x]:
            Du_i.append(
                torch.autograd.grad(u[..., i].sum(), vari, create_graph=True)[0]
            )
        Du_rows.append(torch.cat(Du_i, dim=-1))
    Du = torch.stack(Du_rows, dim=-2)
    return Du
'''

def _coord_desc(d):
    """'x,y' for d=2, 'x,y,z' for d=3, 'x1,...,xd' for d>3."""
    return ",".join("xyz"[:d]) if d <= 3 else ",".join(f"x{i+1}" for i in range(d))

# ─────────────────────────────────────────────────────────────────────────────
# Parametric template builders  — one per operator group
# Each returns a factory(idx, variant, N, hidden) -> (name, code).
# ─────────────────────────────────────────────────────────────────────────────

def _laplacian_template(d):
    f"""Laplacian  \Delta u  in {d}d spatial dimensions."""
    xd = _coord_desc(d)

    def factory(idx, variant, N, hidden):
        name = f"{idx:04d}_L1Laplacian_{d}d_{variant}"
        code = f"""# Problem : L1Laplacian_{d}d
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: laplacian (L11)  —  \Delta u({xd}) = 0
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

{_LAPLACIAN_BLOCK}

class Model(nn.Module):
    \"\"\"
    Laplacian of a network \Delta u with respect to the given variable ({d}D).
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net({d}, {hidden}, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Args:
            x (torch.Tensor):  Input tensor of shape (N, {d}) of the variables in which respect 
            the derivatives have to be computed 
            
        Returns:
            torch.Tensor: \Delta u tensor of shape (N, 1), where every row contains the value of the sum 
            of the second derivatives (laplace) w.r.t the row of the input variable
        \"\"\"  

        with torch.enable_grad():
            x = x.requires_grad_(True)
            u = self.net(x)
            # tp.laplacian (differentialoperators.py:11) 
            return _laplacian(u, x)     


def make_torchphysics_ref(model: Model):
    \"\"\"Return an nn.Module that computes \Delta u via torchphysics ({d}D).\"\"\"
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

    factory.__name__ = f"_laplacian{d}d_template"
    return factory


def _grad_template(d):
    f"""Gradient  \\nabla u  in {d}d spatial dimensions."""
    xd = _coord_desc(d)

    def factory(idx, variant, N, hidden):
        name = f"{idx:04d}_L1Grad_{d}d_{variant}"
        code = f"""# Problem : L1Grad_{d}d
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: grad (L47)  —  \\nabla u({xd})
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

{_GRAD_BLOCK}

class Model(nn.Module):
    \"\"\"
    Gradient of a network \\nabla u with respect to the given variable ({d}D).
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net({d}, {hidden}, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Args:
            x (torch.Tensor):  Input tensor of shape (N, {d}) of the variables in which respect 
            the derivatives have to be computed
            
        Returns:
            torch.Tensor: \\nabla u tensor of shape (N, {d}), where every row contains the values of 
            the first derivatives (gradient) w.r.t the row of the input variable 
        \"\"\"
        
        with torch.enable_grad():
            x = x.requires_grad_(True)
            u = self.net(x)
            #  tp.grad (differentialoperators.py:47) 
            return _grad(u, x)


def make_torchphysics_ref(model: Model):
    \"\"\"Return an nn.Module that computes \\nabla u via torchphysics ({d}D).\"\"\"
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

    factory.__name__ = f"_grad{d}d_template"
    return factory


def _normal_deriv_template(d):
    f"""Normal derivative of a network with respect to the given variable and normal vectors 
    \partial u/\partial n = \\nabla u\cdot n  in {d}d spatial dimensions."""

    def factory(idx, variant, N, hidden):
        name = f"{idx:04d}_L1NormalDeriv_{d}d_{variant}"
        code = f"""# Problem : L1NormalDeriv_{d}d
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: normal_derivative (L111)  —  \partial u / \partial n = \\nabla u \cdot n
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

{_GRAD_BLOCK}

class Model(nn.Module):
    \"\"\"
    Normal derivative of a network with respect to the given variable and normal vectors 
    \partial u / \partial normals = \\nabla u \cdot normals  via autograd ({d}D).
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net({d}, {hidden}, 1)

    def forward(self, x: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Args:
            x (torch.Tensor):  Input tensor of shape (N, {d}) of the variables in which respect 
            the derivatives have to be computed 
            normals (torch.Tensor): The normal vectors of shape (N, {d}) at the points where 
            the derivative has to be computed. In the form: normals = tensor([normal_1, normal_2, ...]
            
        Returns:
            torch.Tensor: \partial u/\partial n tensor of shape (N, 1), where every row contains the values of the normal
            derivatives w.r.t the row of the input variable.
        \"\"\"
 
        with torch.enable_grad():
            x = x.requires_grad_(True)
            u = self.net(x)
            # tp.normal_derivative (differentialoperators.py:111) which calls tp.grad
            gradient = _grad(u, x)
            normal_derivatives = gradient * normals
            return normal_derivatives.sum(dim=-1, keepdim=True)


def make_torchphysics_ref(model: Model):
    \"\"\"Return an nn.Module that computes \partial u/\partial n via torchphysics ({d}D).\"\"\"
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

    factory.__name__ = f"_normal_deriv{d}d_template"
    return factory


def _div_template(d):
    f"""Divergence of a network with respect to the given variable. Only for vector valued inputs, 
    for matices use the function matrix_div. \\nabla\cdot u in {d}d spatial dimensions."""
    xd = _coord_desc(d)

    def factory(idx, variant, N, hidden):
        name = f"{idx:04d}_L1Div_{d}d_{variant}"
        code = f"""# Problem : L1Div_{d}d
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: div (L137)  —  \\nabla \cdot u({xd})
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

{_DIV_BLOCK}

class Model(nn.Module):
    \"\"\"
    Divergence of a network with respect to the given variable. Only for vector valued inputs, 
    for matices use the function matrix_div. \\nabla \cdot u  via autograd ({d}D).
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net({d}, {hidden}, {d})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Args:
            x (torch.Tensor):  Input tensor of shape (N, {d}) of the variables in which respect 
            the derivatives have to be computed. Have to be in a consistent ordering, if for example 
            the output is u = (u_x, u_y) than the variables has to passed in the order (x, y)
            
        Returns:
            torch.Tensor: \\nabla\cdot u tensor of shape (N, 1), where every row contains the values 
            of the divergence of the model w.r.t the row of the input variable.
        \"\"\"    

        with torch.enable_grad():
            x = x.requires_grad_(True)
            u = self.net(x)

            # tp.div (differentialoperators.py:137) 
            return _div(u, x)


def make_torchphysics_ref(model: Model):
    \"\"\"Return an nn.Module that computes \\nabla\cdot u via torchphysics ({d}D).\"\"\"
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

    factory.__name__ = f"_div{d}d_template"
    return factory


def _jac_template(d):
    f"""Jacobian J(u) of a network output with respect to the given input in {d}d spatial dimensions."""
    xd = _coord_desc(d)

    def factory(idx, variant, N, hidden):
        name = f"{idx:04d}_L1Jac_{d}d_{variant}"
        code = f"""# Problem : L1Jac_{d}d
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: jac (L233)  —  J(u)({xd}), u:(N,{d})
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

{_JAC_BLOCK}

class Model(nn.Module):
    \"\"\"
    Jacobian  J(u) of a network output with respect to the given input via autograd ({d}D).
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net({d}, {hidden}, {d})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Args:
            x (torch.Tensor):  Input tensor of shape (N, {d}) in which respect the jacobian should be computed.
            
        Returns:
            torch.Tensor:  J(u) tensor of shape (N, {d}, {d}), where every row contains a jacobian.
        \"\"\"       

        with torch.enable_grad():
            x = x.requires_grad_(True)
            u = self.net(x)

            # tp.jac (differentialoperators.py:233) 
            return _jac(u, x)


def make_torchphysics_ref(model: Model):
    \"\"\"Return an nn.Module that computes J(u) via torchphysics ({d}D).\"\"\"
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

    factory.__name__ = f"_jac{d}d_template"
    return factory


def _rot3d_template(idx, variant, N, hidden):
    name = f"{idx:04d}_L1Rot_3d_{variant}"
    code = f"""# Problem : L1Rot_3d
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: rot (L261)  —  \\nabla \times u(x,y,z), u:(N,3)
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

{_JAC_BLOCK}

class Model(nn.Module):
    \"\"\"
    Rotation / curl \\nabla \times u of a 3-dimensional vector field  (given by a network output) with respect to 
    the given input via autograd (3D).
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net(3, {hidden}, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Args:
            x (torch.Tensor):  Input tensor of shape (N, 3) in which respect the rotation should be
        computed.
            
        Returns:
            torch.Tensor: curl tensor of shape (N, 3), where every row contains a rotation/curl vector 
            for a given batch element.
        \"\"\"      
 
        with torch.enable_grad():
            x = x.requires_grad_(True)
            u = self.net(x)

            jacobian = _jac(u, x)
            rotation = torch.zeros((*(jacobian.shape[:-2]), 3))
            rotation[..., 0] = jacobian[..., 2, 1] - jacobian[..., 1, 2]
            rotation[..., 1] = jacobian[..., 0, 2] - jacobian[..., 2, 0]
            rotation[..., 2] = jacobian[..., 1, 0] - jacobian[..., 0, 1]
            return rotation
          

def make_torchphysics_ref(model: Model):
    \"\"\"Return an nn.Module that computes \\nabla\times u via torchphysics (3D).\"\"\"
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
    f"""The (n-th, possibly mixed) \partial u/\partial t partial derivative of a network output with
    respect to the given variables with {d}d spatial dimensions."""

    def factory(idx, variant, N, hidden):
        name = f"{idx:04d}_L1Partial_{d}dt_{variant}"
        code = f"""# Problem : L1Partial_{d}dt
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: partial (L294)  —  \partial u/\partial t, spatial dim={d}
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


class Model(nn.Module):
    \"\"\"
    The (n-th, possibly mixed) \partial u / \partial t partial derivative of a network output with
    respect to the given variables via autograd ({d}D+t).
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net({d + 1}, {hidden}, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Args:
            x (torch.Tensor):  Input tensor of shape (N, {d})
            t (torch.Tensor):  Input tensors of shape (N, 1) in which respect the derivatives should be computed. If n
        tensors are given, the n-th (mixed) derivative will be computed.
            
        Returns:
            torch.Tensor: tensor \partial u/\partial t of shape (N, 1), where every row contains the values 
            of the computed partial derivative of the model w.r.t the row of the input variable.
        \"\"\"       

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
    \"\"\"Return an nn.Module that computes \partial u / \partial t via torchphysics ({d}D+t).\"\"\"
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

    factory.__name__ = f"_partial{d}dt_template"
    return factory


def _convective_template(d):
    """Convective term :math:`(v \\cdot \\nabla)u` that appears e.g. in material derivatives. 
    Note: This is not the whole material derivative  in d spatial dimensions."""

    def factory(idx, variant, N, hidden):
        name = f"{idx:04d}_L1Convective_{d}d_{variant}"
        code = f"""# Problem : L1Convective_{d}d
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: convective (L320)  —  (v \cdot \\nabla)u, self-advection in {d}D
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


{_JAC_BLOCK}

class Model(nn.Module):
    \"\"\"
    Convective term :math:`(v \\cdot \\nabla)u` that appears e.g. in material derivatives via autograd ({d}D).
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net({d}, {hidden}, {d})

    def forward(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Args:
            u (torch.Tensor):  The vector or scalar field :math:`u` that is convected and should be differentiated.
            x (torch.Tensor):  Input tensor of shape (N, {d}), the spatial variable in which respect u should be differentiated. 
            v (torch.Tensor):  Input tensors of shape (N, {d}), the flow vector field :math:`v`. Should have the same dimension as x.

        Returns:
            torch.Tensor:  A vector or scalar (+batch-dimension) Tensor, that contains the convective derivative.
        \"\"\"       

        with torch.enable_grad():
            x = x.requires_grad_(True)
            u = self.net(x)

            # tp.partial (differentialoperators.py:320) 
            jac_x = _jac(u, x)          
            return torch.bmm(jac_x, v.unsqueeze(dim=2)).squeeze(dim=2) 


def make_torchphysics_ref(model: Model):
    \"\"\"Return an nn.Module that computes (u\cdot\\nabla)u via torchphysics ({d}D).\"\"\"
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

    factory.__name__ = f"_convective{d}d_template"
    return factory


def _sym_grad_template(d):
    """Symmetric gradient  :math:`0.5(\\nabla u + \\nabla u^T) in d spatial dimensions."""

    def factory(idx, variant, N, hidden):
        name = f"{idx:04d}_L1SymGrad_{d}d_{variant}"
        code = f"""# Problem : L1SymGrad_{d}d
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: sym_grad (L345)  —  \varepsilon(u) = 1/2(\\nabla u + \\nabla u^T) in {d}D
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

{_JAC_BLOCK}

class Model(nn.Module):
    \"\"\"
    Symmetric gradient  :math:`0.5(\\nabla u + \\nabla u^T)  via autograd ({d}D).
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net({d}, {hidden}, {d})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Args:
            u (torch.Tensor):  The vector field :math:`u` that should be differentiated.
            x (torch.Tensor):  Input tensor of shape (N, {d}), the spatial variable in which respect u should be differentiated. 

        Returns:
            torch.Tensor:  A Tensor of matrices of the form (N, dim, dim), containing the
            symmetric gradient.
        \"\"\"      

        with torch.enable_grad():
            x = x.requires_grad_(True)
            u = self.net(x)

            # tp.partial (differentialoperators.py:345) 
            jac_matrix = _jac(u, x)
            return 0.5 * (jac_matrix + torch.transpose(jac_matrix, -2, -1)) 


def make_torchphysics_ref(model: Model):
    \"\"\"Return an nn.Module that computes \varepsilon(u) via torchphysics ({d}D).\"\"\"
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

    factory.__name__ = f"_sym_grad{d}d_template"
    return factory


def _matrix_div_template(d):
    """Matrix divergence  \\nabla \cdot \sigma  in d spatial dimensions."""
    out_dim = d * d

    def factory(idx, variant, N, hidden):
        name = f"{idx:04d}_L1MatrixDiv_{d}d_{variant}"
        code = f"""# Problem : L1MatrixDiv_{d}d
# Variant : {variant}  (N={N}, hidden={hidden})
# Operator: matrix_div (L365)  —  \\nabla \cdot \sigma, \sigma:(N,{d},{d}) in {d}D
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

{_DIV_BLOCK}

class Model(nn.Module):
    \"\"\"
    Matrix divergence  \\nabla \cdot \sigma  via autograd ({d}D).

    Net outputs \sigma_flat:(N,{out_dim}) reshaped to \sigma:(N,{d},{d}).
    \"\"\"

{_NET_BLOCK}
    def __init__(self):
        super().__init__()
        self.net = self._build_net({d}, {hidden}, {out_dim})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Args:
            u (torch.Tensor):  The (batch) of matirces that should be differentiated.
            x (torch.Tensor):  Input tensor of shape (N, {d}), the spatial variable in which respect u should be differentiated. 

        Returns:
            torch.Tensor:  A Tensor of vectors of the form (N, dim), containing the
            divegrence of the input.
        \"\"\"      
        with torch.enable_grad():
            x = x.requires_grad_(True)
            sigma = self.net(x).view(-1, {d}, {d})

            # tp.partial (differentialoperators.py:365) 
            div_out = torch.zeros((len(sigma), sigma.shape[1]), device=sigma.device)
            for i in range(sigma.shape[1]):
                # compute divergence of matrix by computing the divergence
                # for each row
                current_row = sigma.narrow(1, i, 1).squeeze(1)
                div_out[:, i : i + 1] = _div(current_row, x)  
            return div_out         


def make_torchphysics_ref(model: Model):
    \"\"\"Return an nn.Module that computes \\nabla\cdot\sigma via torchphysics ({d}D).\"\"\"
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

    factory.__name__ = f"_matrix_div{d}d_template"
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
    # Laplacian  \Delta u 
    (_laplacian2d_template, "small",  128,  32),
    (_laplacian2d_template, "medium", 512,  64),
    (_laplacian2d_template, "large",  1024, 128),
    (_laplacian3d_template, "small",  128,  32),
    (_laplacian3d_template, "medium", 512,  64),
    (_laplacian3d_template, "large",  1024, 128),
    (_laplacian4d_template, "small",  128,  32),
    (_laplacian4d_template, "medium", 512,  64),
    (_laplacian4d_template, "large",  1024, 128),
    # Gradient  \\nabla u 
    (_grad2d_template, "small",  128,  32),
    (_grad2d_template, "medium", 512,  64),
    (_grad2d_template, "large",  1024, 128),
    (_grad3d_template, "small",  128,  32),
    (_grad3d_template, "medium", 512,  64),
    (_grad3d_template, "large",  1024, 128),
    (_grad4d_template, "small",  128,  32),
    (_grad4d_template, "medium", 512,  64),
    (_grad4d_template, "large",  1024, 128),    
    # Normal derivative  \partial u / \partial n 
    (_normal_deriv2d_template, "small",  128,  32),
    (_normal_deriv2d_template, "medium", 512,  64),
    (_normal_deriv2d_template, "large",  1024, 128),
    (_normal_deriv3d_template, "small",  128,  32),
    (_normal_deriv3d_template, "medium", 512,  64),
    (_normal_deriv3d_template, "large",  1024, 128),
    (_normal_deriv4d_template, "small",  128,  32),
    (_normal_deriv4d_template, "medium", 512,  64),
    (_normal_deriv4d_template, "large",  1024, 128),    
    # Divergence  \\nabla \cdot u 
    (_div2d_template, "small",  128,  32),
    (_div2d_template, "medium", 512,  64),
    (_div2d_template, "large",  1024, 128),
    (_div3d_template, "small",  128,  32),
    (_div3d_template, "medium", 512,  64),
    (_div3d_template, "large",  1024, 128),
    (_div4d_template, "small",  128,  32),
    (_div4d_template, "medium", 512,  64),
    (_div4d_template, "large",  1024, 128),    
    # Jacobian  J(u) 
    (_jac2d_template, "small",  128,  32),
    (_jac2d_template, "medium", 512,  64),
    (_jac2d_template, "large",  1024, 128),
    (_jac3d_template, "small",  128,  32),
    (_jac3d_template, "medium", 512,  64),
    (_jac3d_template, "large",  1024, 128),
    (_jac4d_template, "small",  128,  32),
    (_jac4d_template, "medium", 512,  64),
    (_jac4d_template, "large",  1024, 128),    
    # Rotation / curl  \\nabla \times u 
    (_rot3d_template, "small",  128,  32),
    (_rot3d_template, "medium", 512,  64),
    (_rot3d_template, "large",  1024, 128),
    # Partial derivative  \partial u / \partial t 
    (_partial1dt_template, "small",  128,  32),
    (_partial1dt_template, "medium", 512,  64),
    (_partial1dt_template, "large",  1024, 128),
    (_partial2dt_template, "small",  128,  32),
    (_partial2dt_template, "medium", 512,  64),
    (_partial2dt_template, "large",  1024, 128),
    (_partial3dt_template, "small",  128,  32),
    (_partial3dt_template, "medium", 512,  64),
    (_partial3dt_template, "large",  1024, 128),
    # Convective  (u \cdot \\nabla) u 
    (_convective2d_template, "small",  128,  32),
    (_convective2d_template, "medium", 512,  64),
    (_convective2d_template, "large",  1024, 128),
    (_convective3d_template, "small",  128,  32),
    (_convective3d_template, "medium", 512,  64),
    (_convective3d_template, "large",  1024, 128),
    (_convective4d_template, "small",  128,  32),
    (_convective4d_template, "medium", 512,  64),
    (_convective4d_template, "large",  1024, 128),    
    # Symmetric gradient  1/2(\\nabla u + \\nabla u^T) 
    (_sym_grad2d_template, "small",  128,  32),
    (_sym_grad2d_template, "medium", 512,  64),
    (_sym_grad2d_template, "large",  1024, 128),
    (_sym_grad3d_template, "small",  128,  32),
    (_sym_grad3d_template, "medium", 512,  64),
    (_sym_grad3d_template, "large",  1024, 128),
    (_sym_grad4d_template, "small",  128,  32),
    (_sym_grad4d_template, "medium", 512,  64),
    (_sym_grad4d_template, "large",  1024, 128),    
    # Matrix divergence  \\nabla \cdot \sigma 
    (_matrix_div2d_template, "small",  128,  32),
    (_matrix_div2d_template, "medium", 512,  64),
    (_matrix_div2d_template, "large",  1024, 128),
    (_matrix_div3d_template, "small",  128,  32),
    (_matrix_div3d_template, "medium", 512,  64),
    (_matrix_div3d_template, "large",  1024, 128),
    (_matrix_div4d_template, "small",  128,  32),
    (_matrix_div4d_template, "medium", 512,  64),
    (_matrix_div4d_template, "large",  1024, 128),    
]


OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "ScientificKernelBench"
    / "level1"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate():
    for old in OUT_DIR.glob("[0-9]*.py"):
        old.unlink()

    for idx, (factory, variant, N, hidden) in enumerate(TASKS, start=1):
        filename, code = factory(idx, variant, N, hidden)

        refs = _TP_L1_REFS.get(factory, [])
        ref_block = _BUILD_NET_REF + "\n" + "\n".join(
            f"# torchphysics_ref : {r}" for r in refs
        )
        code = ref_block + "\n" + code

        path = OUT_DIR / f"{filename}.py"
        path.write_text(code)
        print(f"  wrote {path.name}")

    print(f"\nGenerated {len(TASKS)} tasks in {OUT_DIR}")


if __name__ == "__main__":
    generate()
