import time
import logging
from typing import Dict, Any, Optional

from app.core.phases.common import (
    EvalStage,
    set_eval_stage,
    make_timing,
    load_reference_code,
    summarize_elapsed_times,
)
from app.core.utils.performance import run_performance
from app.integration import get_reference_path, get_dataset

logger = logging.getLogger(__name__)


def run_baseline_phase(
    backend,
    function: str,
    hardware: str,
    compute_capability: str,
    batch_size=None,
    dim=None,
    input_dims=None,
    torch_compile: bool = False,
    num_trials: Optional[int] = None,
    num_warmup: Optional[int] = None,
) -> Dict[str, Any]:
    set_eval_stage(EvalStage.BASELINE)
    t0 = time.time()
    try:
        backend.cleanup()
        ref_src = load_reference_code(function, batch_size, dim, input_dims)
        exec(ref_src, backend.context)
        elapsed_times = run_performance(
            backend,
            "Model",
            num_trials=num_trials,
            num_warmup=num_warmup,
            torch_compile=torch_compile,
        )
        baseline = summarize_elapsed_times(elapsed_times)
        baseline["device"] = hardware
        baseline["compute_capability"] = compute_capability
        baseline["torch_compile"] = torch_compile
        if batch_size is not None:
            baseline["batch_size"] = batch_size
        if dim is not None:
            baseline["dim"] = dim
        if input_dims:
            baseline["input_dims"] = input_dims
        return {
            "hardware": hardware,
            "compute_capability": compute_capability,
            "compiled": True,
            "baseline": baseline,
            "timing": make_timing(total=time.time() - t0),
        }
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)}"
        return {
            "hardware": hardware,
            "compute_capability": compute_capability,
            "compiled": True,
            "system_error": msg,
            "timing": make_timing(total=time.time() - t0),
        }


def compute_baseline(
    function: str,
    language: str,
    batch_size=None,
    dim=None,
    input_dims=None,
    torch_compile: bool = False,
    num_trials: Optional[int] = None,
    num_warmup: Optional[int] = None,
) -> Dict[str, Any]:
    from app.core.backends.backend_registry import get_backend

    backend = get_backend(language)
    if backend is None:
        return {"error": f"Backend {language} not found"}

    hardware = backend.get_hardware_name()
    capability = backend.get_compute_capability()
    compute_capability = f"{capability[0]}.{capability[1]}" if capability else None

    if function not in get_dataset():
        return {
            "not_supported": True,
            "error": f"Function '{function}' not found in dataset",
            "error_type": "KeyError",
            "device": hardware,
            "compute_capability": compute_capability,
        }

    try:
        result = run_baseline_phase(
            backend,
            function,
            hardware,
            compute_capability,
            batch_size=batch_size,
            dim=dim,
            input_dims=input_dims,
            torch_compile=torch_compile,
            num_trials=num_trials,
            num_warmup=num_warmup,
        )
        if result.get("system_error"):
            return {
                "not_supported": True,
                "error": result.get("system_error"),
                "error_type": "RuntimeError",
                "device": hardware,
                "compute_capability": compute_capability,
            }
        return result["baseline"]
    except Exception as e:
        logger.error(f"Baseline computation failed: {e}", exc_info=True)
        return {
            "not_supported": True,
            "error": str(e),
            "error_type": type(e).__name__,
            "device": hardware,
            "compute_capability": compute_capability,
        }
    finally:
        try:
            backend.cleanup()
        except Exception:
            pass


def compute_all_baselines(language: str, torch_compile: bool = False) -> Dict[str, Any]:
    from app.core.backends.backend_registry import get_backend

    backend = get_backend(language)
    if not backend:
        return {}

    hardware = backend.get_hardware_name()
    capability = backend.get_compute_capability()
    compute_capability = f"{capability[0]}.{capability[1]}" if capability else None
    dataset = get_dataset()

    result = {}
    for op in dataset.keys():
        logger.info(f"Computing baseline for {op}")
        try:
            ref_src_path = get_reference_path(op)
            if not ref_src_path:
                raise FileNotFoundError(f"Reference file not found for function: {op}")
            with open(ref_src_path, "r") as f:
                ref_src = f.read()
            exec(ref_src, backend.context)
            elapsed_times = run_performance(
                backend, "Model", torch_compile=torch_compile
            )
            result[op] = {
                **summarize_elapsed_times(elapsed_times),
                "device": hardware,
                "compute_capability": compute_capability,
                "torch_compile": torch_compile,
            }
        except Exception as e:
            logger.error(f"{op}: {e}")
            result[op] = {
                "not_supported": True,
                "error": str(e),
                "error_type": type(e).__name__,
                "device": hardware,
                "compute_capability": compute_capability,
            }
        backend.clear_device_memory()

    return result
