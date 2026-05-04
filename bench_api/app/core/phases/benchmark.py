import time
import signal
import logging
from typing import Optional, Tuple

import app.config as app_config
from app.core.phases.common import (
    EvalStage,
    set_eval_stage,
    make_timing,
    compile_kernel,
    load_reference_code,
    summarize_elapsed_times,
)
from app.core.phases.results import BenchmarkResult
from app.core.utils.performance import run_performance
from app.core.phases.common import InfraError, raise_if_infra_error

logger = logging.getLogger(__name__)


def _setup_timeout(
    baseline_mean_ms: Optional[float],
    num_trials: int,
    num_warmup: int,
) -> Tuple[Optional[float], bool]:
    if baseline_mean_ms is None:
        return None, False
    # Budget for a kernel up to 50x slower than baseline, plus 10s overhead.
    # The old formula (budget = baseline speed) caused guaranteed timeouts for
    # any kernel slower than baseline with 100 trials.
    worst_case_ms = baseline_mean_ms * 50
    timeout = max(30.0, (num_trials * worst_case_ms + num_warmup * worst_case_ms) / 1000.0 + 10.0)
    return timeout, True


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
        raise_if_infra_error(e)
        logger.error(f"Performance measurement failed: {e}", exc_info=True)
        return None, str(e)
    finally:
        if use_timeout and timeout_seconds is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def _probe_and_adapt(
    backend,
    torch_compile: bool,
    baseline_mean_ms: float,
    default_trials: int,
    default_warmup: int,
) -> Tuple[int, int, str]:
    """
    Run a cheap 2-warmup + 1-trial probe to decide profiling mode.

    Returns (num_trials, num_warmup, mode_label) where mode_label is
    "fast" (kernel is >=5x slower than baseline) or "full" (normal).
    Falls back to full mode on any error.
    """
    try:
        probe_times = run_performance(
            backend, "ModelNew",
            num_trials=1, num_warmup=2,
            torch_compile=torch_compile,
        )
        probe_ms = probe_times[0] if probe_times else None
    except Exception:
        return default_trials, default_warmup, "full"

    if probe_ms is None or probe_ms > 5.0 * baseline_mean_ms:
        return 20, 2, "fast"
    return default_trials, default_warmup, "full"


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
) -> BenchmarkResult:
    set_eval_stage(EvalStage.COMPILATION)

    try:
        ref_src = load_reference_code(function, batch_size, dim, input_dims)
        exec(ref_src, backend.context)
    except Exception as e:
        raise_if_infra_error(e)
        logger.warning(f"[Benchmark] Failed to load reference code for {function}: {e}")

    compile_kernel(backend, function_code, function)
    backend.clear_device_memory()
    t0 = time.time()

    actual_num_trials = (
        num_trials if num_trials is not None else app_config.NUM_PERF_TRIALS
    )
    actual_num_warmup = num_warmup if num_warmup is not None else app_config.NUM_WARMUP

    # Adaptive profiling mode: probe kernel speed, then choose trial count.
    # Only activates when the caller hasn't specified explicit trial counts
    # and a baseline is available for comparison.
    profiling_mode = "full"
    if num_trials is None and num_warmup is None and baseline_mean_ms is not None:
        actual_num_trials, actual_num_warmup, profiling_mode = _probe_and_adapt(
            backend, torch_compile, baseline_mean_ms,
            actual_num_trials, actual_num_warmup,
        )

    logger.info(
        f"[Benchmark] {function}: profiling_mode={profiling_mode} "
        f"trials={actual_num_trials} warmup={actual_num_warmup} "
        f"baseline={f'{baseline_mean_ms:.3g}ms' if baseline_mean_ms else 'N/A'}"
    )

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

    # CUDA profiling: run torch.profiler if timing succeeded
    cuda_profile = None
    if elapsed_times is not None:
        try:
            from app.core.utils.cuda_profiler import profile_kernel
            cuda_profile = profile_kernel(backend, "ModelNew")
        except Exception as e:
            logger.warning(f"[Benchmark] CUDA profiling failed for {function}: {e}")

    return BenchmarkResult(
        hardware=hardware,
        compute_capability=compute_capability,
        performance=summarize_elapsed_times(elapsed_times) if elapsed_times else None,
        performance_info=performance_info,
        timing=make_timing(perf=perf_time, total=time.time() - t0),
        profiling_mode=profiling_mode,
        cuda_profile=cuda_profile,
    )
