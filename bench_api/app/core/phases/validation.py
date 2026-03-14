import time
import logging
from typing import Dict, Any

from app.core.phases.common import (
    EvalStage, set_eval_stage, make_timing, compile_kernel, load_reference_code,
)
from app.core.utils.correctness import run_correctness

logger = logging.getLogger(__name__)


def run_validation_phase(
    backend, function_code: str, function: str, hardware: str, compute_capability: str,
    batch_size=None, dim=None, input_dims=None,
) -> Dict[str, Any]:
    base = {'hardware': hardware, 'compute_capability': compute_capability, 'performance': None}
    t0 = time.time()

    set_eval_stage(EvalStage.COMPILATION)
    t = time.time()
    try:
        compiled, compile_info = compile_kernel(backend, function_code, function)
        if not compiled:
            return {**base, 'compiled': False, 'correctness': None, 'error': compile_info,
                    'stage': EvalStage.COMPILATION, 'timing': make_timing(total=time.time()-t0)}
    except Exception as e:
        msg = f"Re-compilation failed: {str(e)}"
        return {**base, 'compiled': False, 'correctness': None, 'error': msg,
                'stage': EvalStage.COMPILATION, 'timing': make_timing(total=time.time()-t0)}
    comp_time = time.time() - t

    set_eval_stage(EvalStage.CORRECTNESS)
    t = time.time()
    try:
        ref_src = load_reference_code(function, batch_size, dim, input_dims)
        correctness, correctness_info = run_correctness(backend, ref_src)
        backend.clear_device_memory()
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)}"
        return {**base, 'compiled': True, 'correctness': False, 'correctness_info': msg, 'error': msg,
                'stage': EvalStage.CORRECTNESS,
                'timing': make_timing(comp=comp_time, corr=time.time()-t, total=time.time()-t0)}
    corr_time = time.time() - t

    result = {
        **base, 'compiled': True, 'correctness': correctness,
        'stage': EvalStage.CORRECTNESS,
        'timing': make_timing(comp=comp_time, corr=corr_time, total=time.time()-t0),
    }
    if not correctness:
        result['correctness_info'] = correctness_info
        result['error'] = correctness_info or "Correctness check failed"
        if "CUDA error" in result['error'] or "illegal memory access" in result['error'].lower():
            try:
                backend.cleanup()
            except Exception:
                pass
    return result
