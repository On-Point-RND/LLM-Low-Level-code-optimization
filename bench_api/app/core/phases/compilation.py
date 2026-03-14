import time
import logging
from typing import Dict, Any

from app.core.phases.common import EvalStage, set_eval_stage, make_timing, compile_kernel, InfraError

logger = logging.getLogger(__name__)


def run_compilation_phase(
    backend, function_code: str, function: str, hardware: str, compute_capability: str,
) -> Dict[str, Any]:
    base = {'hardware': hardware, 'compute_capability': compute_capability, 'performance': None, 'correctness': None}
    set_eval_stage(EvalStage.COMPILATION)
    t0 = t = time.time()
    try:
        compiled, compile_info = compile_kernel(backend, function_code, function)
    except InfraError:
        raise
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)}"
        return {**base, 'compiled': False, 'compile_info': msg,
                'stage': EvalStage.COMPILATION, 'timing': make_timing(comp=time.time()-t, total=time.time()-t0)}
    comp_time = time.time() - t
    if not compiled:
        return {**base, 'compiled': False, 'compile_info': compile_info,
                'stage': EvalStage.COMPILATION, 'timing': make_timing(comp=comp_time, total=time.time()-t0)}
    return {**base, 'compiled': True, 'stage': EvalStage.COMPILATION,
            'timing': make_timing(comp=comp_time, total=time.time()-t0)}
