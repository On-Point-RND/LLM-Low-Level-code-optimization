import time
import signal
import logging
import torch
from typing import Dict, Any, Optional, Tuple

import app.config as app_config
from app.core.phases.common import (
    EvalStage,
    set_eval_stage,
    make_timing,
    compile_kernel,
    load_reference_code,
    summarize_elapsed_times,
)
from app.core.utils.performance import run_performance
from app.core.phases.common import InfraError

logger = logging.getLogger(__name__)


def _setup_timeout(
    baseline_mean_ms: Optional[float],
    num_trials: int,
    num_warmup: int,
) -> Tuple[Optional[float], bool]:
    if baseline_mean_ms is None:
        return None, False
    total_ms = (num_trials + num_warmup) * baseline_mean_ms
    return max(5.0, (2.0 * total_ms) / 1000.0), True


def _run_timed(
    backend,
    timeout_seconds: Optional[float],
    use_timeout: bool,
    num_trials: int,
    num_warmup: int,
    torch_compile: bool,
) -> Tuple[Optional[list], Optional[str]]:
    old_handler = None

    def timeout_handler(signum, frame):
        raise TimeoutError(
            f"Performance measurement timed out after {timeout_seconds:.2f} seconds"
        )

    try:
        if use_timeout and timeout_seconds is not None:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds) + 1)

        return run_performance(
            backend, "ModelNew", num_trials, num_warmup, torch_compile
        ), None

    except InfraError:
        raise
    except TimeoutError as e:
        logger.error(f"Performance measurement timed out: {e}")
        return None, str(e)
    except Exception as e:
        logger.error(f"Performance measurement failed: {e}", exc_info=True)
        return None, str(e)
    finally:
        if use_timeout and timeout_seconds is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def run_benchmark_phase(
    backend,
    function_code: str,
    function: str,
    hardware: str,
    compute_capability: str,
    batch_size=None,
    dim=None,
    input_dims=None,
    torch_compile: bool = False,
    num_trials: Optional[int] = None,
    num_warmup: Optional[int] = None,
    baseline_mean_ms: Optional[float] = None,
) -> Dict[str, Any]:
    set_eval_stage(EvalStage.COMPILATION)

    try:
        ref_src = load_reference_code(function, batch_size, dim, input_dims)
        exec(ref_src, backend.context)
        model = backend.context.pop("Model", None)
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"[Benchmark] Failed to load reference code for {function}: {e}")

    compile_kernel(backend, function_code, function)
    backend.clear_device_memory()
    t0 = time.time()

    actual_num_trials = (
        num_trials if num_trials is not None else app_config.NUM_PERF_TRIALS
    )
    actual_num_warmup = num_warmup if num_warmup is not None else app_config.NUM_WARMUP
    timeout_seconds, use_timeout = _setup_timeout(
        baseline_mean_ms, actual_num_trials, actual_num_warmup
    )

    set_eval_stage(EvalStage.PERFORMANCE)
    t = time.time()
    elapsed_times, performance_info = _run_timed(
        backend,
        timeout_seconds,
        use_timeout,
        actual_num_trials,
        actual_num_warmup,
        torch_compile,
    )
    perf_time = time.time() - t

    backend.clear_device_memory()

    result = {
        "hardware": hardware,
        "compute_capability": compute_capability,
        "compiled": True,
        "correctness": None,
        "stage": EvalStage.PERFORMANCE,
        "timing": make_timing(perf=perf_time, total=time.time() - t0),
    }
    if performance_info:
        result["performance"] = None
        result["performance_info"] = performance_info
    else:
        result["performance"] = summarize_elapsed_times(elapsed_times)
    return result
