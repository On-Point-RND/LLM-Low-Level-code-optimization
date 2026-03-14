import time
import logging

from app.core.phases.common import (
    EvalStage,
    set_eval_stage,
    make_timing,
    compile_kernel,
    load_reference_code,
    InfraError,
)
from app.core.utils.correctness import run_correctness
from app.core.phases.results import ValidationResult

logger = logging.getLogger(__name__)


def run_validation_phase(
    backend,
    function_code: str,
    function: str,
    hardware: str,
    compute_capability: str,
    batch_size=None,
    dim=None,
    input_dims=None,
) -> ValidationResult:
    t0 = time.time()

    set_eval_stage(EvalStage.COMPILATION)
    t = time.time()
    try:
        compiled, compile_info = compile_kernel(backend, function_code, function)
        if not compiled:
            return ValidationResult(
                hardware=hardware,
                compiled=False,
                correctness=None,
                compute_capability=compute_capability,
                compile_info=compile_info,
                timing=make_timing(total=time.time() - t0),
            )
    except InfraError:
        raise
    except Exception as e:
        return ValidationResult(
            hardware=hardware,
            compiled=False,
            correctness=None,
            compute_capability=compute_capability,
            compile_info=f"Re-compilation failed: {str(e)}",
            timing=make_timing(total=time.time() - t0),
        )
    comp_time = time.time() - t

    set_eval_stage(EvalStage.CORRECTNESS)
    t = time.time()
    try:
        ref_src = load_reference_code(function, batch_size, dim, input_dims)
        correctness, correctness_info = run_correctness(backend, ref_src)
        backend.clear_device_memory()
    except InfraError:
        raise
    except Exception as e:
        return ValidationResult(
            hardware=hardware,
            compiled=True,
            correctness=False,
            compute_capability=compute_capability,
            correctness_info=f"{type(e).__name__}: {str(e)}",
            timing=make_timing(
                comp=comp_time, corr=time.time() - t, total=time.time() - t0
            ),
        )
    corr_time = time.time() - t

    if not correctness:
        info = correctness_info or "Correctness check failed"
        if backend.is_fatal_error(info):
            try:
                backend.cleanup()
            except Exception:
                pass
        return ValidationResult(
            hardware=hardware,
            compiled=True,
            correctness=False,
            compute_capability=compute_capability,
            correctness_info=info,
            timing=make_timing(comp=comp_time, corr=corr_time, total=time.time() - t0),
        )

    return ValidationResult(
        hardware=hardware,
        compiled=True,
        correctness=True,
        compute_capability=compute_capability,
        timing=make_timing(comp=comp_time, corr=corr_time, total=time.time() - t0),
    )
