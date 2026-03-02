import os
import json
import tempfile
import importlib.util
import yaml
import numpy as np
import torch
import tvm
from tvm import relax, dlight as dl
from tvm.ir.transform import PassContext
from tvm.meta_schedule import relax_integration as ms_relax
from tvm.meta_schedule.builder import LocalBuilder
from tvm.meta_schedule.database import MemoryDatabase
from tvm.relax.transform import MetaScheduleApplyDatabase
from .converter import pytorch_to_onnx, pytorch_to_relax, onnx_to_relax, apply_relax_transforms
from .profiler import profile_tir, profile_executable
from .validator import validate_correctness, validate_correctness_executable
from .llm_transform_ir import llm_transform_ir
from .llm_transform_tir import llm_transform_tir
from .schemas import OpenRouterConfig

TARGET = (
    "cuda"
    " -max_threads_per_block=1024"
    " -max_shared_memory_per_block=49152"
    " -thread_warp_size=32"
)


def load_baseline_module(baseline_path):
    spec = importlib.util.spec_from_file_location("baseline", baseline_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    init_inputs = module.get_init_inputs()
    model = module.Model(*init_inputs)
    model.eval()
    inputs = module.get_inputs()
    return model, tuple(inputs)


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_llm_config(config, mode_name):
    llm_dict = config.get("llm_config")
    if not llm_dict:
        raise ValueError(f"llm_config is required when using {mode_name}")
    return OpenRouterConfig(**llm_dict)


def _tune_database(mod, work_dir, max_trials, builder_workers):
    builder = LocalBuilder() if builder_workers == -1 else LocalBuilder(max_workers=builder_workers)
    return ms_relax.tune_relax(
        mod=mod, params={}, target=TARGET,
        work_dir=work_dir, max_trials_global=max_trials,
        builder=builder,
    )



def _apply_default_schedule(mod):
    # Reduction() падает на rfactor/bind (HingeLoss, RMSNorm и др.) — используем GeneralReduction
    return dl.ApplyDefaultSchedule(
        dl.gpu.Matmul(), dl.gpu.GEMV(),
        dl.gpu.GeneralReduction(), dl.gpu.Fallback(),
    )(mod)



def _build_executable(mod, target_obj):
    # MetaSchedule не покрывает все PrimFunc; нескеджуленные остаются и build падает.
    # DefaultGPUSchedule — fallback для elementwise (power и т.п.), уже затюненные не трогает.
    default_tir = tvm.tir.get_default_tir_pipeline(target_obj)
    tir_pipeline = tvm.ir.transform.Sequential([
        tvm.tir.transform.DefaultGPUSchedule(),
        default_tir,
    ])
    return relax.build(mod, target=target_obj, tir_pipeline=tir_pipeline)


def _save_llm_logs(res_dir, llm_log, subdir="llm_logs"):
    if not llm_log:
        return
    llm_dir = os.path.join(res_dir, subdir)
    os.makedirs(llm_dir, exist_ok=True)
    for name, key in [
        ("system_prompt.txt", "system_prompt"),
        ("user_prompt.txt", "user_prompt"),
        ("original_content.txt", "original_content"),
        ("llm_response.txt", "transformed_content"),
    ]:
        with open(os.path.join(llm_dir, name), "w") as f:
            f.write(llm_log[key])
    if "load_error" in llm_log:
        with open(os.path.join(llm_dir, "load_error.txt"), "w") as f:
            f.write(llm_log["load_error"])
    if "parse_error" in llm_log:
        with open(os.path.join(llm_dir, "parse_error.txt"), "w") as f:
            f.write(llm_log["parse_error"])
    log_data = {
        "model": llm_log["model"], "url": llm_log["url"],
        "request_data": llm_log["request_data"],
        "response_data": llm_log["llm_response"],
    }
    if "load_error" in llm_log:
        log_data["load_error"] = llm_log["load_error"]
    if "parse_error" in llm_log:
        log_data["parse_error"] = llm_log["parse_error"]
    with open(os.path.join(llm_dir, "llm_log.json"), "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"      LLM logs saved to: {llm_dir}/")
    if "parse_error" in llm_log or "load_error" in llm_log:
        raise RuntimeError(f"LLM transform failed: {llm_log.get('parse_error') or llm_log.get('load_error')}")


def _llm_metrics(llm_log):
    m = {
        "model": llm_log["model"], "url": llm_log["url"],
        "system_prompt_length": len(llm_log["system_prompt"]),
        "user_prompt_length": len(llm_log["user_prompt"]),
        "original_content_length": len(llm_log["original_content"]),
        "transformed_content_length": len(llm_log["transformed_content"]),
        "usage": llm_log["llm_response"].get("usage"),
        "failed": "load_error" in llm_log or "parse_error" in llm_log,
    }
    if "load_error" in llm_log:
        m["load_error"] = llm_log["load_error"]
    if "parse_error" in llm_log:
        m["parse_error"] = llm_log["parse_error"]
    return m


def _save_ir(mod, out_dir, label, save_json=False):
    os.makedirs(out_dir, exist_ok=True)
    if save_json:
        with open(os.path.join(out_dir, f"{label}_module.json"), "w") as f:
            f.write(tvm.ir.save_json(mod))
    with open(os.path.join(out_dir, f"{label}_module.py"), "w") as f:
        f.write(mod.script())
    print(f"      Saved to: {out_dir}/")


def _run_relax_transform(mod, config, relax_transforms):
    print("  [5/10] Applying Relax transforms...")
    for i, name in enumerate(relax_transforms, 1):
        print(f"      [{i}/{len(relax_transforms)}] {name}")
    return apply_relax_transforms(mod, relax_transforms), None


def _run_llm_ir(mod, config):
    print("  [5/10] Applying LLM transformation (Python IR)...")
    llm_config = _get_llm_config(config, "llm_transform_ir")
    return llm_transform_ir(mod, config=llm_config)


def _run_metaschedule(mod, work_dir, config):
    os.makedirs(work_dir, exist_ok=True)
    max_trials = config.get("meta_schedule_max_trials_global", 256)
    builder_workers = config.get("meta_schedule_builder_workers", -1)
    target_obj = tvm.target.Target(TARGET)
    database = None
    try:
        database = _tune_database(mod, work_dir, max_trials, builder_workers)
    except ValueError as e:
        if "No tasks to tune" not in str(e):
            raise
        print("      No tunable tasks found, building without MetaSchedule tuning...")
    if database is not None:
        with target_obj, database, PassContext(opt_level=3):
            mod = MetaScheduleApplyDatabase(enable_warning=False)(mod)
    # Не запускать DLight после MetaSchedule — он падает на tuned reduction TIR
    return _build_executable(mod, target_obj)


def process_baseline(baseline_path, config_path, kernelbench_root, res_root, gpu_id=0):
    print("  [1/10] Loading configuration...")
    config = load_config(config_path)
    relax_transforms = config.get("relax_transforms", [])

    model_path_rel = os.path.relpath(baseline_path, kernelbench_root).replace(".py", "")
    category = os.path.dirname(model_path_rel) or "."
    name = os.path.basename(model_path_rel) or model_path_rel
    res_dir = os.path.join(res_root, model_path_rel)
    os.makedirs(res_dir, exist_ok=True)

    print("  [2/10] Loading PyTorch model...")
    torch_model, inputs_torch = load_baseline_module(baseline_path)
    print(f"      Model: {name}")
    inp0 = inputs_torch[0]
    print(f"      Input shape: {inp0.shape if isinstance(inp0, torch.Tensor) else 'N/A'}")

    dev = tvm.cuda(gpu_id)
    inputs_tvm = []
    for inp in inputs_torch:
        arr = inp.detach().cpu().numpy() if isinstance(inp, torch.Tensor) else np.array(inp)
        inputs_tvm.append(tvm.runtime.tensor(arr, dev))
    input_shapes = [tuple(x.shape) if hasattr(x, "shape") else None for x in inputs_torch]
    metrics = {
        "model_path": model_path_rel, "category": category, "name": name,
        "target": TARGET, "gpu_id": gpu_id, "input_shapes": input_shapes,
    }

    print("  [3/10] Converting PyTorch to ONNX...")
    onnx_export_error = None
    onnx_path = None
    if config.get("save_onnx", False):
        onnx_dir = os.path.join(res_dir, "onnx")
        os.makedirs(onnx_dir, exist_ok=True)
        onnx_path = os.path.join(onnx_dir, "model.onnx")
        try:
            pytorch_to_onnx(torch_model, inputs_torch, onnx_path)
            print(f"      Saved ONNX to: {onnx_path}")
        except Exception as e:
            onnx_export_error = e
    else:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            onnx_path = tmp.name
        try:
            pytorch_to_onnx(torch_model, inputs_torch, onnx_path)
            print("      ONNX created (temporary)")
        except Exception as e:
            onnx_export_error = e
            os.unlink(onnx_path)
            onnx_path = None

    print("  [4/10] Converting ONNX to Relax IR...")
    if onnx_export_error is not None:
        print(f"      ONNX export failed ({onnx_export_error.__class__.__name__}), falling back to torch.fx...")
        mod = pytorch_to_relax(torch_model, inputs_torch)
    else:
        try:
            mod = onnx_to_relax(onnx_path, keep_params_in_input=False)
        except Exception as e:
            print(f"      ONNX→Relax failed ({e.__class__.__name__}), falling back to torch.fx...")
            mod = pytorch_to_relax(torch_model, inputs_torch)
        finally:
            if not config.get("save_onnx", False) and onnx_path and os.path.exists(onnx_path):
                os.unlink(onnx_path)

    transform_relax_mode = config.get("transform_relax_mode", "relax_transforms")
    transform_tir_mode = config.get("transform_tir_mode", "tir_transforms")
    relax_handlers = {
        "relax_transforms": lambda m: _run_relax_transform(m, config, relax_transforms),
        "llm_transform_ir": lambda m: _run_llm_ir(m, config),
    }
    if transform_relax_mode not in relax_handlers:
        raise ValueError(f"Unknown transform_relax_mode: {transform_relax_mode}")
    mod, llm_log = relax_handlers[transform_relax_mode](mod)

    if llm_log:
        _save_llm_logs(res_dir, llm_log, "llm_logs")
        metrics["llm_transform"] = _llm_metrics(llm_log)

    if config.get("save_relax_ir", False):
        print("  [6/10] Saving Relax IR...")
        _save_ir(mod, os.path.join(res_dir, "relax_ir"), "relax", save_json=config.get("save_relax_json", False))

    mod_executable = None
    if transform_tir_mode == "llm_transform_tir":
        print("  [7/10] Applying LLM transformation (TIR / TVMScript)...")
        llm_config = _get_llm_config(config, "llm_transform_tir")
        mod, llm_log_tir = llm_transform_tir(mod, config=llm_config)
        if llm_log_tir:
            _save_llm_logs(res_dir, llm_log_tir, "llm_logs_tir")
            metrics["llm_transform_tir"] = _llm_metrics(llm_log_tir)
    else:
        print("  [7/10] Tuning with MetaSchedule (Relax)...")
        ms_work_dir = os.path.join(res_root, model_path_rel, "meta_schedule")
        mod_executable = _run_metaschedule(mod, ms_work_dir, config)

    if config.get("profile_after_tir", False):
        print("  [8/10] Profiling after TIR transforms / MetaSchedule (GPU)...", flush=True)
        dev.sync()
        prof_cfg = config.get("profile", {})
        warmup = prof_cfg.get("warmup_iters", 3)
        number = prof_cfg.get("number", 10)
        repeat = prof_cfg.get("repeat", 5)
        if transform_tir_mode == "llm_transform_tir":
            prof_result = profile_tir(mod, inputs_tvm, dev, target=TARGET, warmup_iters=warmup, number=number, repeat=repeat)
        else:
            prof_result = profile_executable(
                mod_executable, inputs_tvm, dev,
                skip_prof_report=config.get("skip_prof_report", True),
                warmup_iters=warmup, number=number, repeat=repeat,
                verbose=prof_cfg.get("verbose", True),
            )
        print(f"      Mean latency: {prof_result['mean_ms']:.3f} ms", flush=True)
        profiles_dir = os.path.join(res_dir, "profiles")
        os.makedirs(profiles_dir, exist_ok=True)
        pr = prof_result.get("profiler_report")
        if pr is not None:
            with open(os.path.join(profiles_dir, "after_tir.txt"), "w") as f:
                f.write(pr.table())
            with open(os.path.join(profiles_dir, "after_tir.json"), "w") as f:
                f.write(pr.json())
        if "latency" not in metrics:
            metrics["latency"] = {}
        metrics["latency"]["after_tir"] = {
            k: prof_result[k]
            for k in ("mean_ms", "std_ms", "min_ms", "max_ms", "median_ms", "p95_ms", "p99_ms")
        }

    if config.get("save_tir_ir", False):
        print("  [9/10] Saving TIR IR...")
        _save_ir(mod, os.path.join(res_dir, "tir_ir"), "tir", save_json=config.get("save_tir_json", False))

    if config.get("check_correctness", False):
        print("  [10/10] Checking correctness...")
        if transform_tir_mode == "llm_transform_tir":
            correctness = validate_correctness(torch_model, mod, inputs_torch, inputs_tvm, dev)
        else:
            correctness = validate_correctness_executable(torch_model, mod_executable, inputs_torch, inputs_tvm, dev)
        metrics["correctness"] = correctness
        print(f"      Correctness: {'✓' if correctness['is_correct'] else '✗'} (max_abs_diff: {correctness['max_abs_diff']:.2e})")

    metrics_file = os.path.join(res_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"      Metrics saved to: {metrics_file}")
    return metrics
