import sys
import os
import torch
import numpy as np
import time
import signal
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from app.config import MULTIKERNELBENCH_PATH, REFERENCE_DIR
import app.config as app_config
from app.integration import get_reference_path, get_dataset

from app.core.backends.backend_registry import get_backend
from app.core.utils.code_utils import extract_first_code
from app.core.utils.correctness import run_correctness
from app.core.utils.performance import run_performance

logger = logging.getLogger(__name__)


class EvalStage:
    INITIALIZATION = "initialization"
    BASELINE = "baseline"
    COMPILATION = "compilation"
    CORRECTNESS = "correctness"
    PERFORMANCE = "performance"


_current_eval_stage = EvalStage.INITIALIZATION


def get_current_eval_stage():
    return _current_eval_stage


def _compile_kernel_code(function_code: str, function: str, backend) -> Tuple[bool, str]:
    generated_code = extract_first_code(function_code, ['python', 'cpp'])
    if generated_code is None:
        generated_code = function_code

    compiled, compile_info = backend.compile(generated_code, function)

    if not compiled:
        if compile_info:
            logger.debug(f"Compilation failed for {function}: {compile_info}")
        return False, compile_info or "Compilation failed"

    return True, ""


def _override_dimensions_in_code(ref_src: str, batch_size: Optional[int], dim: Optional[int], input_dims: Optional[Dict[str, Any]], function: str) -> str:
    dimension_overrides = []
    if batch_size is not None:
        dimension_overrides.append(f"batch_size = {batch_size}")
    if dim is not None:
        dimension_overrides.append(f"dim = {dim}")
    if input_dims:
        for key, value in input_dims.items():
            dimension_overrides.append(f"{key} = {value}")

    if dimension_overrides:
        lines = ref_src.split('\n')
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                insert_pos = i + 1
            elif line.strip().startswith('class ') and insert_pos > 0:
                break

        override_code = '\n'.join(dimension_overrides) + '\n'
        ref_src = '\n'.join(lines[:insert_pos]) + '\n' + override_code + '\n'.join(lines[insert_pos:])

    return ref_src


def _do_compilation(backend, function_code, function):
    backend.clear_device_memory()
    compiled, compile_info = _compile_kernel_code(function_code, function, backend)
    return compiled, compile_info


def _do_correctness_check(backend, function, batch_size, dim, input_dims):
    ref_src_path = get_reference_path(function)
    if not ref_src_path:
        raise FileNotFoundError(f"Reference file not found for function: {function}")

    with open(ref_src_path, 'r') as f:
        ref_src = f.read()

    ref_src = _override_dimensions_in_code(ref_src, batch_size, dim, input_dims, function)
    correctness, correctness_info = run_correctness(backend, ref_src)
    backend.clear_device_memory()
    return correctness, correctness_info


def _do_baseline_computation(function, language, batch_size, dim, input_dims):
    baseline_result = compute_baseline(function, language, batch_size, dim, input_dims)

    if isinstance(baseline_result, dict) and 'mean' in baseline_result:
        return baseline_result['mean']
    return None


def _do_performance_measurement(backend, baseline_mean_ms, function, num_trials, torch_compile):
    actual_num_trials = num_trials if num_trials is not None else app_config.NUM_PERF_TRIALS
    timeout_seconds, use_timeout = _setup_performance_measurement_timeout(baseline_mean_ms, actual_num_trials)
    elapsed_times, performance_error = _execute_performance_measurement_with_timeout(
        backend, timeout_seconds, use_timeout, function, actual_num_trials, torch_compile
    )
    backend.clear_device_memory()
    return elapsed_times, performance_error


def _make_timing(comp=0.0, corr=0.0, perf=0.0, total=0.0):
    return {'compilation': comp, 'correctness': corr, 'performance': perf, 'total': total}


def _do_validation(
    backend, function_code, function, language, hardware, compute_capability,
    batch_size, dim, input_dims,
) -> Dict[str, Any]:
    global _current_eval_stage
    base = {'hardware': hardware, 'compute_capability': compute_capability, 'performance': None}

    _current_eval_stage = EvalStage.BASELINE
    try:
        baseline_mean_ms = _do_baseline_computation(function, language, batch_size, dim, input_dims)
    except Exception as e:
        return {**base, 'compiled': False, 'correctness': None,
                'stage': EvalStage.BASELINE, 'error': f"Baseline computation failed: {str(e)}"}

    t0 = time.time()

    _current_eval_stage = EvalStage.COMPILATION
    t = time.time()
    try:
        compiled, compile_info = _do_compilation(backend, function_code, function)
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)}"
        return {**base, 'compiled': False, 'correctness': None, 'compile_info': msg, 'error': msg,
                'stage': EvalStage.COMPILATION, 'timing': _make_timing(comp=time.time()-t, total=time.time()-t0)}
    comp_time = time.time() - t

    if not compiled:
        return {**base, 'compiled': False, 'correctness': None,
                'compile_info': compile_info, 'error': compile_info,
                'stage': EvalStage.COMPILATION, 'timing': _make_timing(comp=comp_time, total=time.time()-t0)}

    _current_eval_stage = EvalStage.CORRECTNESS
    t = time.time()
    try:
        correctness, correctness_info = _do_correctness_check(backend, function, batch_size, dim, input_dims)
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)}"
        return {**base, 'compiled': True, 'correctness': False, 'correctness_info': msg, 'error': msg,
                'stage': EvalStage.CORRECTNESS,
                'timing': _make_timing(comp=comp_time, corr=time.time()-t, total=time.time()-t0)}
    corr_time = time.time() - t

    result = {
        **base, 'compiled': True, 'correctness': correctness,
        'stage': EvalStage.CORRECTNESS,
        'timing': _make_timing(comp=comp_time, corr=corr_time, total=time.time()-t0),
        'baseline_mean_ms': baseline_mean_ms,
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


def _do_benchmark(
    backend, function_code, function, language, hardware, compute_capability,
    batch_size, dim, input_dims, torch_compile, num_trials, baseline_mean_ms,
) -> Dict[str, Any]:
    global _current_eval_stage
    _current_eval_stage = EvalStage.COMPILATION

    # Load reference code so get_inputs/get_init_inputs are in context for performance measurement.
    try:
        ref_src_path = get_reference_path(function)
        if ref_src_path:
            with open(ref_src_path, 'r') as f:
                ref_src = f.read()
            ref_src = _override_dimensions_in_code(ref_src, batch_size, dim, input_dims, function)
            exec(ref_src, backend.context)
    except Exception as e:
        logger.warning(f"[Benchmark] Failed to load reference code for {function}: {e}")

    _do_compilation(backend, function_code, function)
    t0 = time.time()

    _current_eval_stage = EvalStage.PERFORMANCE
    t = time.time()
    try:
        elapsed_times, performance_error = _do_performance_measurement(
            backend, baseline_mean_ms, function, num_trials, torch_compile
        )
    except Exception as e:
        elapsed_times, performance_error = None, str(e)
    perf_time = time.time() - t

    result = {
        'hardware': hardware, 'compute_capability': compute_capability,
        'compiled': True, 'correctness': None,
        'stage': EvalStage.PERFORMANCE,
        'timing': _make_timing(perf=perf_time, total=time.time()-t0),
    }
    if performance_error:
        result['performance'] = None
        result['performance_error'] = performance_error
        result['error'] = performance_error
    elif elapsed_times:
        result['performance'] = {
            "mean": float(f"{np.mean(elapsed_times):.3g}"),
            "std": float(f"{np.std(elapsed_times):.3g}"),
            "min": float(f"{np.min(elapsed_times):.3g}"),
            "max": float(f"{np.max(elapsed_times):.3g}"),
            "num_trials": len(elapsed_times),
        }
    else:
        result['performance'] = None
    return result


def _do_kernel_evaluation(
    backend, function_code, function, language, hardware, torch_compile, num_trials,
    batch_size, dim, input_dims,
    mode: str = 'validation',
    baseline_mean_ms: Optional[float] = None,
) -> Dict[str, Any]:
    capability = backend.get_compute_capability()
    compute_capability = f"{capability[0]}.{capability[1]}" if capability else None
    if mode == 'benchmark':
        return _do_benchmark(
            backend, function_code, function, language, hardware, compute_capability,
            batch_size, dim, input_dims, torch_compile, num_trials, baseline_mean_ms,
        )
    return _do_validation(
        backend, function_code, function, language, hardware, compute_capability,
        batch_size, dim, input_dims,
    )


def evaluate_kernel(
    function_code: str,
    function: str,
    language: str,
    torch_compile: bool = False,
    num_trials: Optional[int] = None,
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None,
    mode: str = 'validation',
    baseline_mean_ms: Optional[float] = None,
) -> Dict[str, Any]:
    backend = None
    hardware = "unknown"
    compute_capability = None

    try:
        backend = get_backend(language)
        if backend is None:
            return {
                'compiled': False,
                'correctness': None,
                'performance': None,
                'hardware': hardware,
                'compute_capability': compute_capability,
                'error': f"Backend for {language} not found"
            }

        hardware = backend.get_hardware_name()
        capability = backend.get_compute_capability()
        compute_capability = f"{capability[0]}.{capability[1]}" if capability else None

        logger.debug(f"Starting evaluation for {function} on {language} (mode={mode})")
        return _do_kernel_evaluation(
            backend, function_code, function, language, hardware, torch_compile, num_trials,
            batch_size, dim, input_dims, mode=mode, baseline_mean_ms=baseline_mean_ms,
        )

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Evaluation failed: {error_msg}", exc_info=True)
        return {
            'compiled': False,
            'correctness': None,
            'performance': None,
            'hardware': hardware,
            'compute_capability': compute_capability,
            'stage': _current_eval_stage,
            'error': error_msg
        }

    finally:
        try:
            if backend:
                backend.cleanup()
        except Exception:
            pass


def _prepare_backend_for_baseline(backend):
    backend.cleanup()


def _load_reference_code_and_override_dims(function: str, batch_size: Optional[int], dim: Optional[int], input_dims: Optional[Dict[str, Any]]) -> str:
    ref_src_path = get_reference_path(function)
    if not ref_src_path:
        error_msg = f"Reference file not found for function: {function}"
        logger.error(f"{function}: {error_msg}")
        raise FileNotFoundError(error_msg)

    with open(ref_src_path, 'r') as f:
        ref_src = f.read()

    ref_src = _override_dimensions_in_code(ref_src, batch_size, dim, input_dims, function)
    return ref_src


def _execute_baseline_with_error_handling(backend, function: str, hardware: str) -> Dict[str, Any]:
    elapsed_times = run_performance(backend, 'Model')
    return {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
        'device': hardware
    }


def compute_baseline(
    function: str,
    language: str,
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    backend = get_backend(language)
    if backend is None:
        return {"error": f"Backend {language} not found"}

    hardware = backend.get_hardware_name()
    capability = backend.get_compute_capability()
    compute_capability = f"{capability[0]}.{capability[1]}" if capability else None
    dataset = get_dataset()

    if function not in dataset:
        return {
            "not_supported": True,
            "error": f"Function '{function}' not found in dataset",
            "error_type": "KeyError",
            "device": hardware,
            "compute_capability": compute_capability
        }

    try:
        _prepare_backend_for_baseline(backend)
        ref_src = _load_reference_code_and_override_dims(function, batch_size, dim, input_dims)
        exec(ref_src, backend.context)

        result = _execute_baseline_with_error_handling(backend, function, hardware)
        result['compute_capability'] = compute_capability

        if batch_size is not None:
            result['batch_size'] = batch_size
        if dim is not None:
            result['dim'] = dim
        if input_dims:
            result['input_dims'] = input_dims

        return result

    except Exception as e:
        logger.error(f"Baseline computation failed: {e}", exc_info=True)
        return {
            "not_supported": True,
            "error": str(e),
            "error_type": type(e).__name__,
            "device": hardware,
            "compute_capability": compute_capability
        }

    finally:
        try:
            backend.cleanup()
        except Exception:
            pass


def _setup_performance_measurement_timeout(baseline_mean_ms: Optional[float], num_trials: int) -> Tuple[Optional[float], bool]:
    if baseline_mean_ms is None:
        return None, False
    total_baseline_time_ms = (num_trials + app_config.NUM_WARMUP) * baseline_mean_ms
    timeout_seconds = max(5.0, (2.0 * total_baseline_time_ms) / 1000.0)
    return timeout_seconds, True


def _execute_performance_measurement_with_timeout(backend, timeout_seconds, use_timeout, function, num_trials, torch_compile) -> Tuple[Optional[list], Optional[str]]:
    old_handler = None

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Performance measurement timed out after {timeout_seconds:.2f} seconds")

    try:
        if use_timeout and timeout_seconds is not None:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds) + 1)

        elapsed_times = run_performance(backend, 'ModelNew', num_trials, torch_compile)

        if elapsed_times is None:
            logger.warning("Performance measurement returned None")
            return None, None
        if not isinstance(elapsed_times, (list, tuple, np.ndarray)):
            logger.warning("Performance measurement returned unexpected type")
            return None, None

        return elapsed_times, None

    except TimeoutError as e:
        logger.error(f"Performance measurement timed out: {e}")
        return None, str(e)
    except Exception as e:
        logger.error(f"Performance measurement failed: {e}", exc_info=True)
        return None, str(e)

    finally:
        if use_timeout and timeout_seconds is not None:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


def compute_all_baselines(language: str) -> Dict[str, Any]:
    backend = get_backend(language)
    if not backend:
        return {}

    hardware = backend.get_hardware_name()
    capability = backend.get_compute_capability()
    compute_capability = f"{capability[0]}.{capability[1]}" if capability else None
    dataset = get_dataset()

    result = {}

    for op in dataset.keys():
        logger.info(f'Computing baseline for {op}')
        try:
            ref_src_path = get_reference_path(op)
            if not ref_src_path:
                raise FileNotFoundError(f"Reference file not found for function: {op}")

            with open(ref_src_path, 'r') as f:
                ref_src = f.read()

            exec(ref_src, backend.context)
            elapsed_times = run_performance(backend, 'Model')

            result[op] = {
                "mean": float(f"{np.mean(elapsed_times):.3g}"),
                "std": float(f"{np.std(elapsed_times):.3g}"),
                "min": float(f"{np.min(elapsed_times):.3g}"),
                "max": float(f"{np.max(elapsed_times):.3g}"),
                "num_trials": len(elapsed_times),
                'device': hardware,
                'compute_capability': compute_capability
            }

        except Exception as e:
            logger.error(f"{op}: {e}")
            result[op] = {
                "not_supported": True,
                "error": str(e),
                "error_type": type(e).__name__,
                "device": hardware,
                "compute_capability": compute_capability
            }

        backend.clear_device_memory()

    return result
