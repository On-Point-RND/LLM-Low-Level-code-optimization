import time
import logging

from app.core.phases.common import (
    EvalStage,
    set_eval_stage,
    make_timing,
    compile_kernel,
    InfraError,
)
from app.core.phases.results import CompilationResult

logger = logging.getLogger(__name__)


def run_compilation_phase(
    backend,
    function_code: str,
    function: str,
    hardware: str,
    compute_capability: str,
) -> CompilationResult:
    set_eval_stage(EvalStage.COMPILATION)
    t0 = t = time.time()
    try:
        compiled, compile_info = compile_kernel(backend, function_code, function)
    except InfraError:
        raise
    except Exception as e:
        return CompilationResult(
            hardware=hardware,
            compiled=False,
            compute_capability=compute_capability,
            compile_info=f"{type(e).__name__}: {str(e)}",
            timing=make_timing(comp=time.time() - t, total=time.time() - t0),
        )
    comp_time = time.time() - t
    return CompilationResult(
        hardware=hardware,
        compiled=compiled,
        compute_capability=compute_capability,
        compile_info=compile_info or None,
        timing=make_timing(comp=comp_time, total=time.time() - t0),
    )
