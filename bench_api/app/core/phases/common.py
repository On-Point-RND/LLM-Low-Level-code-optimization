import logging
import numpy as np
from typing import Dict, Any, Tuple

from app.integration import get_reference_path


class InfraError(Exception):
    """Raised when infrastructure (our code) fails, as opposed to user kernel failures."""

    pass


_INFRA_ERRNOS = {
    12,  # ENOMEM: Cannot allocate memory (system RAM exhausted)
    28,  # ENOSPC: No space left on device
}


def raise_if_infra_error(e: Exception) -> None:
    """Re-raise as InfraError if the exception is a system-level resource problem."""
    import torch
    if isinstance(e, torch.cuda.OutOfMemoryError):
        raise InfraError(f"GPU out of memory: {e}") from e
    if isinstance(e, OSError) and e.errno in _INFRA_ERRNOS:
        raise InfraError(f"System resource error (errno {e.errno}): {e}") from e
    if isinstance(e, RuntimeError) and "no kernel image is available for execution on the device" in str(e):
        raise InfraError(f"CUDA arch mismatch (check TORCH_CUDA_ARCH_LIST): {e}") from e


logger = logging.getLogger(__name__)


class EvalStage:
    INITIALIZATION = "initialization"
    BASELINE = "baseline"
    COMPILATION = "compilation"
    CORRECTNESS = "correctness"
    PERFORMANCE = "performance"


_current_eval_stage = EvalStage.INITIALIZATION


def get_current_eval_stage() -> str:
    return _current_eval_stage


def set_eval_stage(stage: str):
    global _current_eval_stage
    _current_eval_stage = stage


def make_timing(comp=0.0, corr=0.0, perf=0.0, total=0.0) -> Dict[str, Any]:
    return {
        "compilation": comp,
        "correctness": corr,
        "performance": perf,
        "total": total,
    }


def override_dimensions(ref_src: str, batch_size, dim, input_dims) -> str:
    overrides = []
    if batch_size is not None:
        overrides.append(f"batch_size = {batch_size}")
    if dim is not None:
        overrides.append(f"dim = {dim}")
    if input_dims:
        for key, value in input_dims.items():
            overrides.append(f"{key} = {value}")

    if not overrides:
        return ref_src

    lines = ref_src.split("\n")
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            insert_pos = i + 1
        elif line.strip().startswith("class ") and insert_pos > 0:
            break

    override_code = "\n".join(overrides) + "\n"
    return (
        "\n".join(lines[:insert_pos])
        + "\n"
        + override_code
        + "\n".join(lines[insert_pos:])
    )


def load_reference_code(
    function: str, batch_size=None, dim=None, input_dims=None
) -> str:
    try:
        ref_src_path = get_reference_path(function)
        if not ref_src_path:
            raise FileNotFoundError(
                f"Reference file not found for function: {function}"
            )
        with open(ref_src_path, "r") as f:
            ref_src = f.read()
    except Exception as e:
        raise InfraError(f"Failed to load reference code for {function}: {e}") from e
    return override_dimensions(ref_src, batch_size, dim, input_dims)


def compile_kernel(backend, function_code: str, function: str) -> Tuple[bool, str]:
    from app.core.utils.code_utils import extract_first_code

    generated_code = (
        extract_first_code(function_code, ["python", "cpp"]) or function_code
    )
    try:
        backend.clear_device_memory()
    except Exception as e:
        raise InfraError(f"Failed to clear device memory: {e}") from e
    try:
        compiled, compile_info = backend.compile(generated_code, function)
    except Exception as e:
        raise_if_infra_error(e)
        return False, f"{type(e).__name__}: {e}"
    if not compiled:
        logger.debug(f"Compilation failed for {function}: {compile_info}")
        return False, compile_info or "Compilation failed"
    return True, ""


def summarize_elapsed_times(elapsed_times) -> Dict[str, Any]:
    return {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }
