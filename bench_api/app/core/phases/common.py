import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple

from app.integration import get_reference_path

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
    return {'compilation': comp, 'correctness': corr, 'performance': perf, 'total': total}


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

    lines = ref_src.split('\n')
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            insert_pos = i + 1
        elif line.strip().startswith('class ') and insert_pos > 0:
            break

    override_code = '\n'.join(overrides) + '\n'
    return '\n'.join(lines[:insert_pos]) + '\n' + override_code + '\n'.join(lines[insert_pos:])


def load_reference_code(function: str, batch_size=None, dim=None, input_dims=None) -> str:
    ref_src_path = get_reference_path(function)
    if not ref_src_path:
        raise FileNotFoundError(f"Reference file not found for function: {function}")
    with open(ref_src_path, 'r') as f:
        ref_src = f.read()
    return override_dimensions(ref_src, batch_size, dim, input_dims)


def compile_kernel(backend, function_code: str, function: str) -> Tuple[bool, str]:
    from app.core.utils.code_utils import extract_first_code
    generated_code = extract_first_code(function_code, ['python', 'cpp']) or function_code
    backend.clear_device_memory()
    compiled, compile_info = backend.compile(generated_code, function)
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
