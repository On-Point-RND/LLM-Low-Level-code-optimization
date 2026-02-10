import os
import json
import importlib.util
import yaml
import numpy as np
import torch
import tvm
from tvm import relax
from .converter import pytorch_to_onnx, onnx_to_relax, apply_relax_transforms, apply_tir_transforms
from .profiler import profile_relax, profile_tir
from .validator import validate_correctness
from .llm_transform_ir import llm_transform_ir
from .llm_transform_graph import llm_transform_graph
from .schemas import OpenRouterConfig

TARGET = "cuda"

def load_baseline_module(baseline_path):
    spec = importlib.util.spec_from_file_location("baseline", baseline_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    init_inputs = module.get_init_inputs()
    model = module.Model(*init_inputs)
    model.eval()
    
    inputs = module.get_inputs()
    inputs_torch = tuple(inputs)
    
    return model, inputs_torch

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def process_baseline(baseline_path, config_path, dataset_root, res_root):
    print("  [1/10] Loading configuration...")
    config = load_config(config_path)
    dataset_dir = os.path.dirname(config_path)
    presets_dir = os.path.join(dataset_dir, "presets")
    preset_name = config.get("preset")
    if preset_name:
        preset_path = os.path.join(presets_dir, f"{preset_name}.yaml")
        print(f"      Using preset: {preset_name}")
        with open(preset_path, "r") as f:
            preset_cfg = yaml.safe_load(f)
        relax_transforms = preset_cfg.get("relax_transforms", [])
        tir_transforms = preset_cfg.get("tir_transforms", [])
        print(f"      Relax transforms: {len(relax_transforms)}")
        print(f"      TIR transforms: {len(tir_transforms)}")
    else:
        relax_transforms = config.get("relax_transforms", [])
        tir_transforms = config.get("tir_transforms", [])
    
    model_path_rel = os.path.relpath(baseline_path, dataset_root)
    model_path_rel = model_path_rel.replace('.py', '')
    if os.path.dirname(model_path_rel):
        category = os.path.dirname(model_path_rel)
        name = os.path.basename(model_path_rel)
    else:
        category = "."
        name = model_path_rel
    
    res_dir = os.path.join(res_root, model_path_rel)
    os.makedirs(res_dir, exist_ok=True)
    
    print("  [2/10] Loading PyTorch model...")
    torch_model, inputs_torch = load_baseline_module(baseline_path)
    print(f"      Model: {name}")
    print(f"      Input shape: {inputs_torch[0].shape if isinstance(inputs_torch[0], torch.Tensor) else 'N/A'}")
    
    dev = tvm.cuda(0)
    
    inputs_tvm = []
    for inp in inputs_torch:
        if isinstance(inp, torch.Tensor):
            inputs_tvm.append(tvm.runtime.tensor(inp.detach().cpu().numpy(), dev))
        else:
            inputs_tvm.append(tvm.runtime.tensor(np.array(inp), dev))
    
    metrics = {
        "model_path": model_path_rel,
        "category": category if category else ".",
        "name": name,
        "target": TARGET
    }
    
    print("  [3/10] Converting PyTorch to ONNX...")
    onnx_path = None
    if config.get("save_onnx", False):
        onnx_dir = os.path.join(res_dir, "onnx")
        os.makedirs(onnx_dir, exist_ok=True)
        onnx_path = os.path.join(onnx_dir, "model.onnx")
        pytorch_to_onnx(torch_model, inputs_torch, onnx_path)
        print(f"      Saved ONNX to: {onnx_path}")
    else:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            onnx_path = tmp.name
        pytorch_to_onnx(torch_model, inputs_torch, onnx_path)
        print("      ONNX created (temporary)")
    
    print("  [4/10] Converting ONNX to Relax IR...")
    mod = onnx_to_relax(onnx_path, keep_params_in_input=False)
    
    if not config.get("save_onnx", False):
        os.unlink(onnx_path)
    
    transform_mode = config.get("transform_mode", "relax_transforms")
    llm_log = None
    
    if transform_mode == "relax_transforms":
        print("  [5/10] Applying Relax transforms...")
        for i, transform_name in enumerate(relax_transforms, 1):
            print(f"      [{i}/{len(relax_transforms)}] {transform_name}")
        mod = apply_relax_transforms(mod, relax_transforms)
    elif transform_mode == "llm_transform_ir":
        print("  [5/10] Applying LLM transformation (Python IR)...")
        llm_config_dict = config.get("llm_config")
        if not llm_config_dict:
            raise ValueError("llm_config is required when using llm_transform_ir")
        llm_config = OpenRouterConfig(**llm_config_dict)
        mod, llm_log = llm_transform_ir(mod, config=llm_config)
    elif transform_mode == "llm_transform_graph":
        print("  [5/10] Applying LLM transformation (JSON graph)...")
        llm_config_dict = config.get("llm_config")
        if not llm_config_dict:
            raise ValueError("llm_config is required when using llm_transform_graph")
        llm_config = OpenRouterConfig(**llm_config_dict)
        mod, llm_log = llm_transform_graph(mod, config=llm_config)
    else:
        raise ValueError(f"Unknown transform_mode: {transform_mode}")
    
    if llm_log:
        llm_log_dir = os.path.join(res_dir, "llm_logs")
        os.makedirs(llm_log_dir, exist_ok=True)
        
        with open(os.path.join(llm_log_dir, "system_prompt.txt"), "w") as f:
            f.write(llm_log["system_prompt"])
        
        with open(os.path.join(llm_log_dir, "user_prompt.txt"), "w") as f:
            f.write(llm_log["user_prompt"])
        
        with open(os.path.join(llm_log_dir, "original_content.txt"), "w") as f:
            f.write(llm_log["original_content"])
        
        with open(os.path.join(llm_log_dir, "llm_response.txt"), "w") as f:
            f.write(llm_log["transformed_content"])
        
        log_data = {
            "model": llm_log["model"],
            "url": llm_log["url"],
            "request_data": llm_log["request_data"],
            "response_data": llm_log["llm_response"]
        }
        if "load_error" in llm_log:
            log_data["load_error"] = llm_log["load_error"]
            with open(os.path.join(llm_log_dir, "load_error.txt"), "w") as f:
                f.write(llm_log["load_error"])
        if "parse_error" in llm_log:
            log_data["parse_error"] = llm_log["parse_error"]
            with open(os.path.join(llm_log_dir, "parse_error.txt"), "w") as f:
                f.write(llm_log["parse_error"])
        
        with open(os.path.join(llm_log_dir, "llm_log.json"), "w") as f:
            json.dump(log_data, f, indent=2)
        
        metrics["llm_transform"] = {
            "model": llm_log["model"],
            "url": llm_log["url"],
            "system_prompt_length": len(llm_log["system_prompt"]),
            "user_prompt_length": len(llm_log["user_prompt"]),
            "original_content_length": len(llm_log["original_content"]),
            "transformed_content_length": len(llm_log["transformed_content"]),
            "usage": llm_log["llm_response"].get("usage"),
            "failed": "load_error" in llm_log or "parse_error" in llm_log
        }
        if "load_error" in llm_log:
            metrics["llm_transform"]["load_error"] = llm_log["load_error"]
        if "parse_error" in llm_log:
            metrics["llm_transform"]["parse_error"] = llm_log["parse_error"]
        
        print(f"      LLM logs saved to: {llm_log_dir}/")
    
    if config.get("save_relax_ir", False):
        print("  [6/10] Saving Relax IR...")
        relax_ir_dir = os.path.join(res_dir, "relax_ir")
        os.makedirs(relax_ir_dir, exist_ok=True)
        
        relax_json_file = os.path.join(relax_ir_dir, "relax_module.json")
        json_str = tvm.ir.save_json(mod)
        with open(relax_json_file, "w") as f:
            f.write(json_str)
        
        relax_py_file = os.path.join(relax_ir_dir, "relax_module.py")
        with open(relax_py_file, "w") as f:
            f.write(mod.script())
        print(f"      Saved to: {relax_ir_dir}/")
    
    print("  [7/10] Applying TIR transforms...")
    for i, transform_name in enumerate(tir_transforms, 1):
        print(f"      [{i}/{len(tir_transforms)}] {transform_name}")
    mod = apply_tir_transforms(mod, tir_transforms, target=TARGET)
    
    if config.get("profile_after_tir", False):
        print("  [8/10] Profiling after TIR transforms (GPU)...")
        profile_after_tir = profile_tir(mod, inputs_tvm, dev, target=TARGET)
        print(f"      Mean latency: {profile_after_tir['mean_ms']:.3f} ms")
        
        profiles_dir = os.path.join(res_dir, "profiles")
        os.makedirs(profiles_dir, exist_ok=True)
        
        with open(os.path.join(profiles_dir, "after_tir.txt"), "w") as f:
            f.write(profile_after_tir["profiler_report"].table())
        with open(os.path.join(profiles_dir, "after_tir.json"), "w") as f:
            f.write(profile_after_tir["profiler_report"].json())
        
        if "latency" not in metrics:
            metrics["latency"] = {}
        metrics["latency"]["after_tir"] = {
            "mean_ms": profile_after_tir["mean_ms"],
            "std_ms": profile_after_tir["std_ms"],
            "min_ms": profile_after_tir["min_ms"],
            "max_ms": profile_after_tir["max_ms"],
            "median_ms": profile_after_tir["median_ms"],
            "p95_ms": profile_after_tir["p95_ms"],
            "p99_ms": profile_after_tir["p99_ms"]
        }
    
    if config.get("save_tir_ir", False):
        print("  [9/10] Saving TIR IR...")
        tir_ir_dir = os.path.join(res_dir, "tir_ir")
        os.makedirs(tir_ir_dir, exist_ok=True)
        
        tir_json_file = os.path.join(tir_ir_dir, "tir_module.json")
        tir_json_str = tvm.ir.save_json(mod)
        with open(tir_json_file, "w") as f:
            f.write(tir_json_str)
        
        tir_py_file = os.path.join(tir_ir_dir, "tir_module.py")
        with open(tir_py_file, "w") as f:
            f.write(mod.script())
        print(f"      Saved to: {tir_ir_dir}/")
    
    if config.get("check_correctness", False):
        print("  [10/10] Checking correctness...")
        correctness = validate_correctness(torch_model, mod, inputs_torch, inputs_tvm, dev)
        metrics["correctness"] = correctness
        status = "✓" if correctness["is_correct"] else "✗"
        print(f"      Correctness: {status} (max_abs_diff: {correctness['max_abs_diff']:.2e})")
    
    metrics_file = os.path.join(res_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"      Metrics saved to: {metrics_file}")
    
    return metrics
