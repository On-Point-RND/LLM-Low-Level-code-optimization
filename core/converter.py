import os
import numpy as np
import torch
import onnx
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
import torch.onnx

def pytorch_to_onnx(model, example_inputs, onnx_path, input_names=None, output_names=None):
    model.eval()
    with torch.no_grad():
        if input_names is None:
            input_names = [f"input_{i}" for i in range(len(example_inputs))]
        if output_names is None:
            output_names = ["output"]
        
        # do_constant_folding=False — иначе broadcast/expand могут экспортироваться некорректно (HingeLoss и др.)
        torch.onnx.export(
            model,
            example_inputs if isinstance(example_inputs, tuple) else tuple(example_inputs),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=14,
            do_constant_folding=False,
            dynamic_axes=None
        )
    return onnx_path

def onnx_to_relax(onnx_path, shape_dict=None, dtype_dict="float32", keep_params_in_input=False):
    onnx_model = onnx.load(onnx_path)
    mod = from_onnx(
        onnx_model,
        shape_dict=shape_dict,
        dtype_dict=dtype_dict,
        keep_params_in_input=keep_params_in_input,
        sanitize_input_names=True
    )
    return mod

def apply_relax_transforms(mod, transform_names):
    for transform_name in transform_names:
        transform_func = getattr(relax.transform, transform_name)
        mod = transform_func()(mod)
    return mod

def apply_tir_transforms(mod, transform_names, target="cuda"):
    target_obj = tvm.target.Target(target)
    with target_obj:
        for transform_name in transform_names:
            if transform_name == "DefaultGPUSchedule":
                transform_func = tvm.tir.transform.DefaultGPUSchedule
            else:
                transform_func = getattr(tvm.tir.transform, transform_name)
            mod = transform_func()(mod)
    return mod
