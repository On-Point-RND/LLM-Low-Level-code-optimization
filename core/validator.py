import numpy as np
import torch
import tvm
from tvm import relax

from .build_utils import build_relax_cuda


def _compute_torch_output(torch_model, inputs_torch):
    torch_model.eval()
    with torch.no_grad():
        y_torch = torch_model(*inputs_torch)
        if isinstance(y_torch, tuple):
            y_torch = y_torch[0]
        return y_torch.cpu().numpy()


def _compare_outputs(y_torch_np, y_tvm_np, abs_tolerance, rel_tolerance):
    diff = y_tvm_np - y_torch_np
    max_diff = float(np.max(np.abs(diff)))
    mean_diff = float(np.mean(np.abs(diff)))
    rel_diff = float(np.mean(np.abs(diff) / (np.abs(y_torch_np) + 1e-8)))
    max_rel_diff = float(np.max(np.abs(diff) / (np.abs(y_torch_np) + 1e-8)))
    is_correct = (max_diff < abs_tolerance) or (max_rel_diff < rel_tolerance)
    return {
        "is_correct": is_correct,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "mean_rel_diff": rel_diff,
        "max_rel_diff": max_rel_diff,
        "abs_tolerance": abs_tolerance,
        "rel_tolerance": rel_tolerance,
        "tvm_shape": list(y_tvm_np.shape),
        "torch_shape": list(y_torch_np.shape),
    }


def validate_correctness(
    torch_model,
    tvm_mod,
    inputs_torch,
    inputs_tvm,
    dev,
    func_name="main",
    abs_tolerance=1e-3,
    rel_tolerance=1e-4,
):
    y_torch_np = _compute_torch_output(torch_model, inputs_torch)
    ex = build_relax_cuda(tvm_mod, target="cuda")
    vm = relax.VirtualMachine(ex, dev)
    y_tvm = vm[func_name](*inputs_tvm)
    y_tvm_np = y_tvm.numpy()
    return _compare_outputs(y_torch_np, y_tvm_np, abs_tolerance, rel_tolerance)


def validate_correctness_safe(
    torch_model,
    tvm_mod,
    inputs_torch,
    inputs_tvm,
    dev,
    func_name="main",
    abs_tolerance=1e-3,
    rel_tolerance=1e-4,
):
    """
    Like validate_correctness but catches build/VM errors. Returns
    {is_correct: False, build_error: str} on build/VM failure instead of raising.
    """
    y_torch_np = _compute_torch_output(torch_model, inputs_torch)
    try:
        ex = build_relax_cuda(tvm_mod, target="cuda")
        vm = relax.VirtualMachine(ex, dev)
        y_tvm = vm[func_name](*inputs_tvm)
        y_tvm_np = y_tvm.numpy()
    except Exception as e:
        return {
            "is_correct": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "mean_rel_diff": None,
            "max_rel_diff": None,
            "abs_tolerance": abs_tolerance,
            "rel_tolerance": rel_tolerance,
            "tvm_shape": None,
            "torch_shape": list(y_torch_np.shape),
            "build_error": str(e),
        }
    return _compare_outputs(y_torch_np, y_tvm_np, abs_tolerance, rel_tolerance)


def validate_correctness_executable(
    torch_model,
    executable,
    inputs_torch,
    inputs_tvm,
    dev,
    func_name="main",
    abs_tolerance=1e-3,
    rel_tolerance=1e-4,
):
    y_torch_np = _compute_torch_output(torch_model, inputs_torch)
    vm = relax.VirtualMachine(executable, dev)
    y_tvm = vm[func_name](*inputs_tvm)
    y_tvm_np = y_tvm.numpy()
    return _compare_outputs(y_torch_np, y_tvm_np, abs_tolerance, rel_tolerance)
