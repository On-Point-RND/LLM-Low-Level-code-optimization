import sys
import os
import importlib
import torch
import numpy as np
import time
import traceback
import signal
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from app.config import MULTIKERNELBENCH_PATH, REFERENCE_DIR
import app.config as app_config
from app.integration import get_reference_path, get_dataset

from app.core.backends.backend_registry import get_backend
from app.core.utils.code_utils import extract_first_code

logger = logging.getLogger(__name__)

def _cleanup_cuda():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"Failed to clean CUDA: {e}")


def _compile_kernel_code(function_code: str, function: str, backend, language: str) -> Tuple[bool, str]:
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


def _create_compiled_time_execution(backend, num_trials: Optional[int] = None):
    actual_num_trials = num_trials if num_trials is not None else app_config.NUM_PERF_TRIALS
    
    device = backend.get_device()
    if hasattr(torch.cuda, 'synchronize') and torch.cuda.is_available():
        synchronize = torch.cuda.synchronize
        event_class = torch.cuda.Event
    else:
        def synchronize(device=None): pass
        event_class = None

    get_inputs = backend.context['get_inputs']
    get_init_inputs = backend.context['get_init_inputs']
    ModelNew = backend.context['ModelNew']

    init_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_init_inputs()]
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]

    with torch.no_grad():
        model_instance = ModelNew(*init_inputs).to(device)

        try:
            compiled_model = torch.compile(model_instance)
        except Exception as compile_error:
            error_msg = str(compile_error)
            logger.warning(f"torch.compile failed, using original model: {error_msg}")
            compiled_model = model_instance

        def compiled_time_execution(eval_target='ModelNew'):
            for _ in range(app_config.NUM_WARMUP):
                compiled_model(*inputs)
                synchronize(device=device)

            elapsed_times = []
            if event_class:
                for _ in range(actual_num_trials):
                    start_event = event_class(enable_timing=True)
                    end_event = event_class(enable_timing=True)
                    start_event.record()
                    compiled_model(*inputs)
                    end_event.record()
                    synchronize(device=device)
                    elapsed_times.append(start_event.elapsed_time(end_event))
            else:
                for _ in range(actual_num_trials):
                    synchronize(device=device)
                    start = time.time()
                    compiled_model(*inputs)
                    synchronize(device=device)
                    elapsed_times.append((time.time() - start) * 1000)

            return elapsed_times

    return compiled_time_execution


def _setup_torch_compile(backend, num_trials: Optional[int] = None):
    if not hasattr(torch, 'compile'):
        logger.warning(f"torch.compile is not available in PyTorch {torch.__version__}. torch.compile requires PyTorch 2.0+. Continuing without compilation.")
        return False

    if 'ModelNew' not in backend.context:
        return False

    compiled_time_execution = _create_compiled_time_execution(backend, num_trials)
    backend.time_execution = compiled_time_execution
    return True


def _do_compilation(backend, function_code, function, language):
    if language == 'cuda':
        _cleanup_cuda()

    compiled, compile_info = _compile_kernel_code(function_code, function, backend, language)
    return compiled, compile_info


def _do_correctness_check(backend, function, language, batch_size, dim, input_dims):
    ref_src_path = get_reference_path(function)
    if not ref_src_path:
        raise FileNotFoundError(f"Reference file not found for function: {function}")

    with open(ref_src_path, 'r') as f:
        ref_src = f.read()

    ref_src = _override_dimensions_in_code(ref_src, batch_size, dim, input_dims, function)
    correctness, correctness_info = backend.correctness_execution(ref_src)

    if language == 'cuda':
        _cleanup_cuda()

    return correctness, correctness_info


def _do_baseline_computation(function, language, batch_size, dim, input_dims):
    baseline_result = compute_baseline(function, language, batch_size, dim, input_dims)

    if language == 'cuda':
        _cleanup_cuda()

    if isinstance(baseline_result, dict) and 'mean' in baseline_result:
        return baseline_result['mean']
    return None


def _do_performance_measurement(backend, baseline_mean_ms, function, language, num_trials):
    if num_trials is not None:
        original_config = _override_num_perf_trials(language, num_trials)
    else:
        original_config = None

    actual_num_trials = num_trials if num_trials is not None else app_config.NUM_PERF_TRIALS
    timeout_seconds, use_timeout = _setup_performance_measurement_timeout(baseline_mean_ms, actual_num_trials)
    elapsed_times, performance_error = _execute_performance_measurement_with_timeout(backend, timeout_seconds, use_timeout, function)

    _restore_num_perf_trials(language, original_config)

    if language == 'cuda':
        _cleanup_cuda()

    return elapsed_times, performance_error


def _do_kernel_evaluation(backend, function_code, function, language, hardware, torch_compile, num_trials, batch_size, dim, input_dims) -> Dict[str, Any]:
    capability = backend.get_compute_capability()
    compute_capability = f"{capability[0]}.{capability[1]}" if capability else None
    result = {
        'compiled': False,
        'correctness': None,
        'performance': None,
        'hardware': hardware,
        'compute_capability': compute_capability,
        'timing': {
            'compilation': 0.0,
            'correctness': 0.0,
            'performance': 0.0,
            'total': 0.0
        }
    }

    # Calculate baseline first because it cleans up the backend context
    baseline_mean_ms = None
    try:
        baseline_mean_ms = _do_baseline_computation(function, language, batch_size, dim, input_dims)
    except Exception as e:
        logger.error(f"Baseline computation failed: {e}")
        return {
            'compiled': False,
            'correctness': None,
            'performance': None,
            'hardware': hardware,
            'compute_capability': compute_capability,
            'error': f"Baseline computation failed: {str(e)}"
        }

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    eval_start_time = time.time()
    try:
        try:
            comp_start = time.time()
            compiled, compile_info = _do_compilation(backend, function_code, function, language)
            result['timing']['compilation'] = time.time() - comp_start
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            result['compile_info'] = error_msg
            result['error'] = error_msg
            result['timing']['total'] = time.time() - eval_start_time
            return result

        if not compiled:
            result['compile_info'] = compile_info
            result['error'] = compile_info
            result['timing']['total'] = time.time() - eval_start_time
            return result

        result['compiled'] = True

        try:
            corr_start = time.time()
            correctness, correctness_info = _do_correctness_check(backend, function, language, batch_size, dim, input_dims)
            result['timing']['correctness'] = time.time() - corr_start
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            result['correctness'] = False
            result['correctness_info'] = error_msg
            result['error'] = error_msg
            result['timing']['total'] = time.time() - eval_start_time
            return result

        result['correctness'] = correctness

        if not correctness:
            result['correctness_info'] = correctness_info
            result['error'] = correctness_info or "Correctness check failed"
            if "CUDA error" in result['error'] or "illegal memory access" in result['error'].lower():
                try:
                    backend.cleanup()
                    backend.context = {}
                except Exception:
                    pass
                result['timing']['total'] = time.time() - eval_start_time
                return result
    finally:
        if "CUDA_LAUNCH_BLOCKING" in os.environ:
            del os.environ["CUDA_LAUNCH_BLOCKING"]

    if torch_compile:
        _setup_torch_compile(backend, num_trials=num_trials)

    try:
        perf_start = time.time()
        elapsed_times, performance_error = _do_performance_measurement(backend, baseline_mean_ms, function, language, num_trials)
        result['timing']['performance'] = time.time() - perf_start
    except Exception as e:
        elapsed_times = None
        performance_error = str(e)
        result['timing']['performance'] = time.time() - perf_start

    result['timing']['total'] = time.time() - eval_start_time

    if performance_error:
        result['performance'] = None
        result['performance_error'] = performance_error
        if 'error' not in result:
            result['error'] = performance_error
    elif elapsed_times and len(elapsed_times) > 0:
        result['performance'] = {
            "mean": float(f"{np.mean(elapsed_times):.3g}"),
            "std": float(f"{np.std(elapsed_times):.3g}"),
            "min": float(f"{np.min(elapsed_times):.3g}"),
            "max": float(f"{np.max(elapsed_times):.3g}"),
            "num_trials": len(elapsed_times),
        }

    return result


def evaluate_kernel(
    function_code: str,
    function: str,
    language: str,
    torch_compile: bool = False,
    num_trials: Optional[int] = None,
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None
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

        logger.debug(f"Starting evaluation for {function} on {language}")
        result = _do_kernel_evaluation(backend, function_code, function, language, hardware, torch_compile, num_trials, batch_size, dim, input_dims)
        return result

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Evaluation failed: {error_msg}", exc_info=True)
        return {
            'compiled': False,
            'correctness': None,
            'performance': None,
            'hardware': hardware,
            'compute_capability': compute_capability,
            'error': error_msg
        }

    finally:
        try:
            if language == 'cuda':
                _cleanup_cuda()
        except Exception:
            pass
        try:
            if backend:
                backend.cleanup()
        except Exception:
            pass


def _prepare_backend_for_baseline(backend, language: str):
    if language == 'cuda':
        _cleanup_cuda()
    try:
        backend.cleanup()
    except Exception:
        pass
    if hasattr(backend, 'context'):
        backend.context = {}


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


def _execute_baseline_with_error_handling(backend, function: str, language: str, hardware: str) -> Dict[str, Any]:
    elapsed_times = backend.time_execution('Model')
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
        _prepare_backend_for_baseline(backend, language)
        ref_src = _load_reference_code_and_override_dims(function, batch_size, dim, input_dims)
        
        # We need to manually exec if the backend exposes context, or add a method to backend to load reference code
        # CudaBackend exposes context, so we can exec into it.
        # Ideally, we should add 'load_reference_code' to backend interface, but for now:
        if hasattr(backend, 'context'):
            exec(ref_src, backend.context)
            
        result = _execute_baseline_with_error_handling(backend, function, language, hardware)
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


def _override_num_perf_trials(language: str, num_trials: int) -> Optional[int]:
    original_config = app_config.NUM_PERF_TRIALS
    app_config.NUM_PERF_TRIALS = num_trials
    return original_config


def _restore_num_perf_trials(language: str, original_config: Optional[int]):
    if original_config is not None:
        app_config.NUM_PERF_TRIALS = original_config


def _setup_performance_measurement_timeout(baseline_mean_ms: Optional[float], num_trials: int) -> Tuple[Optional[float], bool]:
    timeout_seconds = None
    use_timeout = False
    if baseline_mean_ms is not None:
        total_baseline_time_ms = (num_trials + app_config.NUM_WARMUP) * baseline_mean_ms
        timeout_seconds = max(5.0, (2.0 * total_baseline_time_ms) / 1000.0)
        use_timeout = True

    return timeout_seconds, use_timeout


def _execute_performance_measurement_with_timeout(backend, timeout_seconds: Optional[float], use_timeout: bool, function: str) -> Tuple[Optional[list], Optional[str]]:
    old_handler = None

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Performance measurement timed out after {timeout_seconds:.2f} seconds")

    try:
        if use_timeout and timeout_seconds is not None:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds) + 1)

        elapsed_times = backend.time_execution()

        if elapsed_times is None:
            logger.warning(f"Performance measurement returned None")
            return None, None
        if not isinstance(elapsed_times, (list, tuple, np.ndarray)):
            logger.warning(f"Performance measurement returned unexpected type")
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
    op_tests = dataset.keys()
    
    for op in op_tests:
        logger.info(f'Computing baseline for {op}')
        try:
            ref_src_path = get_reference_path(op)
            if not ref_src_path:
                raise FileNotFoundError(f"Reference file not found for function: {op}")

            with open(ref_src_path, 'r') as f:
                ref_src = f.read()

            if hasattr(backend, 'context'):
                exec(ref_src, backend.context)
                
            elapsed_times = backend.time_execution('Model')

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

        if language == 'cuda':
            _cleanup_cuda()

    return result