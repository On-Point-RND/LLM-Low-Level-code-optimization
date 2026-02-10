import numpy as np
import torch
import tvm
from tvm import relax

def validate_correctness(torch_model, tvm_mod, inputs_torch, inputs_tvm, dev, func_name="main", abs_tolerance=1e-3, rel_tolerance=1e-4):
    torch_model.eval()
    with torch.no_grad():
        y_torch = torch_model(*inputs_torch)
        if isinstance(y_torch, tuple):
            y_torch = y_torch[0]
        y_torch_np = y_torch.cpu().numpy()
    
    target = tvm.target.Target("cuda")
    ex = relax.build(tvm_mod, target)
    vm = relax.VirtualMachine(ex, dev)
    y_tvm = vm[func_name](*inputs_tvm)
    y_tvm_np = y_tvm.numpy()
    
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
        "torch_shape": list(y_torch_np.shape)
    }
