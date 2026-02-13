import sys
import os
import importlib
import torch
import numpy as np
import time
import traceback
import signal
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from app.config import MULTIKERNELBENCH_PATH, REFERENCE_DIR

sys.path.insert(0, str(MULTIKERNELBENCH_PATH))




def _cleanup_cuda():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"[INFO] Cleaned CUDA cache")
    except Exception as e:
        print(f"[WARNING] Failed to clean CUDA: {e}")

import config as mk_config
import config

ref_path_abs = str(REFERENCE_DIR.resolve())
mk_config.ref_impl_base_path = ref_path_abs
mk_config.project_root_path = str(MULTIKERNELBENCH_PATH.resolve())

if not os.path.exists(ref_path_abs):
    raise RuntimeError(f"Reference directory does not exist: {ref_path_abs}")

config.ref_impl_base_path = ref_path_abs
config.project_root_path = str(MULTIKERNELBENCH_PATH.resolve())

print(f"[INFO] Using REFERENCE_DIR: {ref_path_abs}")

from backends.backend_registry import BACKEND_REGISTRY
from utils.evaluation_utils import extract_first_code
from utils.utils import get_ref_src_path
from utils.performance import time_execution_event_template
from dataset import dataset
from config import num_perf_trials, num_warmup
import utils.performance as performance_module

try:
    test_path = get_ref_src_path('leaky_relu')
    expected_path = os.path.join(ref_path_abs, 'activation', 'leaky_relu.py')
    if test_path != expected_path:
        print(f"[WARNING] get_ref_src_path path mismatch!")
        print(f"[WARNING]   Got: {test_path}")
        print(f"[WARNING]   Expected: {expected_path}")
        print(f"[WARNING]   config.ref_impl_base_path = {config.ref_impl_base_path}")
    if not os.path.exists(test_path):
        print(f"[ERROR] get_ref_src_path returned non-existent path: {test_path}")
        print(f"[ERROR] Expected path should be: {expected_path}")
        print(f"[ERROR] REFERENCE_DIR = {REFERENCE_DIR}")
        print(f"[ERROR] ref_path_abs = {ref_path_abs}")
except Exception as e:
    print(f"[WARNING] Error testing get_ref_src_path: {e}")
    traceback.print_exc()

if not os.path.exists(mk_config.ref_impl_base_path):
    raise RuntimeError(f"Reference directory does not exist: {mk_config.ref_impl_base_path}")


def get_backend(language: str):
    if language not in BACKEND_REGISTRY:
        try:
            importlib.import_module(f"backends.{language}_backend")
        except ImportError as e:
            raise ValueError(f"Unsupported language/platform: {language} (module not found)") from e
    
    backend_instance = BACKEND_REGISTRY.get(language)
    if backend_instance is None:
        raise ValueError(f"Unsupported language/platform: {language}")
    
    backend_class = type(backend_instance)
    backend = backend_class()
    return backend


def _handle_compile_error(backend, language, error):
    error_str = str(error)
    if isinstance(error, AttributeError) and 'context' in error_str.lower():
        error_msg = f"Backend context error during compilation: {error_str}. Backend type: {type(backend)}, has context: {hasattr(backend, 'context')}"
        print(f"[ERROR] {error_msg}")
        return {'compile_info': error_msg}
    if 'context' in error_str.lower():
        error_msg = f"Error during compilation (context-related): {error_str}"
        print(f"[ERROR] {error_msg}")
        return {'compile_info': error_msg}
    return {'compile_info': error_str}


def _compile_kernel_code(function_code: str, function: str, backend, language: str) -> Tuple[bool, str]:
    generated_code = extract_first_code(function_code, ['python', 'cpp'])
    if generated_code is None:
        generated_code = function_code

    print(f"[DEBUG] Attempting to compile code for {function}")
    compiled, compile_info = backend.compile(generated_code, function)
    print(f"[DEBUG] Compilation result for {function}: compiled={compiled}")

    if not compiled:
        if compile_info:
            print(f"[DEBUG] Compilation failed for {function}:")
            print(compile_info)
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
        print(f"[INFO] Overriding dimensions for evaluation {function}: {', '.join(dimension_overrides)}")

    return ref_src


def _create_compiled_time_execution(backend, num_trials: Optional[int] = None):
    actual_num_trials = num_trials if num_trials is not None else num_perf_trials
    OriginalModelNew = backend.context['ModelNew']

    device = backend.get_device()
    if hasattr(torch.cuda, 'synchronize'):
        synchronize = torch.cuda.synchronize
        event_class = torch.cuda.Event
    else:
        def synchronize(device=None):
            pass
        event_class = None

    get_inputs = backend.context['get_inputs']
    get_init_inputs = backend.context['get_init_inputs']
    ModelNew = backend.context['ModelNew']

    init_inputs = get_init_inputs()
    init_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in init_inputs]

    inputs = get_inputs()
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

    with torch.no_grad():
        model_instance = ModelNew(*init_inputs).to(device)

        try:
            compiled_model = torch.compile(model_instance)
        except Exception as compile_error:
            error_msg = str(compile_error)
            if 'pytree' in error_msg.lower() or 'register_pytree_node' in error_msg:
                print(f"[WARNING] torch.compile failed due to PyTorch compatibility issue (PyTorch {torch.__version__}): {error_msg}")
                print(f"[INFO] This may be a compatibility issue. Try using torch_compile=false or check PyTorch installation. Continuing with original model (no compilation).")
            else:
                print(f"[WARNING] torch.compile failed, using original model: {error_msg}")
            compiled_model = model_instance

        for _ in range(num_warmup):
            compiled_model(*inputs)
            synchronize(device=device)

        def compiled_time_execution(eval_target='ModelNew'):
            elapsed_times = []
            if event_class:
                for _ in range(actual_num_trials):
                    start_event = event_class(enable_timing=True)
                    end_event = event_class(enable_timing=True)
                    start_event.record()
                    compiled_model(*inputs)
                    end_event.record()
                    synchronize(device=device)
                    elapsed_time_ms = start_event.elapsed_time(end_event)
                    elapsed_times.append(elapsed_time_ms)
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
        print(f"[WARNING] torch.compile is not available in PyTorch {torch.__version__}. torch.compile requires PyTorch 2.0+. Continuing without compilation.")
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
    ref_src_path = os.path.join(str(REFERENCE_DIR), dataset[function]['category'], f'{function}.py')
    if not os.path.exists(ref_src_path):
        raise FileNotFoundError(f"Reference file not found: {ref_src_path}")

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
        original_config, original_performance, original_pallas = _override_num_perf_trials(language, num_trials)
    else:
        original_config = original_performance = original_pallas = None

    actual_num_trials = num_trials if num_trials is not None else num_perf_trials
    timeout_seconds, use_timeout = _setup_performance_measurement_timeout(baseline_mean_ms, actual_num_trials)
    elapsed_times, performance_error = _execute_performance_measurement_with_timeout(backend, timeout_seconds, use_timeout, function)

    _restore_num_perf_trials(language, original_config, original_performance, original_pallas)

    if language == 'cuda':
        _cleanup_cuda()

    return elapsed_times, performance_error


def _do_kernel_evaluation(backend, function_code, function, language, hardware, torch_compile, num_trials, batch_size, dim, input_dims) -> Dict[str, Any]:
    result = {
        'compiled': False,
        'correctness': None,
        'performance': None,
        'hardware': hardware
    }

    try:
        compiled, compile_info = _do_compilation(backend, function_code, function, language)
    except Exception as e:
        result['compile_info'] = str(e)
        result['error'] = str(e)
        return result

    if not compiled:
        result['compile_info'] = compile_info
        result['error'] = compile_info
        return result

    result['compiled'] = True

    try:
        correctness, correctness_info = _do_correctness_check(backend, function, language, batch_size, dim, input_dims)
    except Exception as e:
        result['correctness'] = False
        result['correctness_info'] = str(e)
        result['error'] = str(e)
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
            return result

    if torch_compile:
        _setup_torch_compile(backend, num_trials=num_trials)

    try:
        baseline_mean_ms = _do_baseline_computation(function, language, batch_size, dim, input_dims)
    except Exception as e:
        baseline_mean_ms = None
        if 'error' not in result:
            result['error'] = str(e)

    try:
        elapsed_times, performance_error = _do_performance_measurement(backend, baseline_mean_ms, function, language, num_trials)
    except Exception as e:
        elapsed_times = None
        performance_error = str(e)

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

    try:
        backend = get_backend(language)
        hardware = backend.get_hardware_name()

        if not hasattr(backend, 'context'):
            error_msg = f"Backend {language} does not have 'context' attribute"
            print(f"[ERROR] {error_msg}")
            traceback.print_exc()
            return {
                'compiled': False,
                'correctness': None,
                'performance': None,
                'hardware': hardware,
                'error': error_msg
            }

        print(f"[DEBUG] Starting evaluation for {function} on {language}")
        result = _do_kernel_evaluation(backend, function_code, function, language, hardware, torch_compile, num_trials, batch_size, dim, input_dims)
        return result

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        traceback.print_exc()
        return {
            'compiled': False,
            'correctness': None,
            'performance': None,
            'hardware': hardware,
            'error': str(e)
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
    backend.context = {}


def _load_reference_code_and_override_dims(function: str, batch_size: Optional[int], dim: Optional[int], input_dims: Optional[Dict[str, Any]]) -> str:
    ref_src_path = os.path.join(str(REFERENCE_DIR), dataset[function]['category'], f'{function}.py')
    if not os.path.exists(ref_src_path):
        error_msg = f"Reference file not found: {ref_src_path} (REFERENCE_DIR={REFERENCE_DIR})"
        print(f"[ERROR] {function}: {error_msg}")
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
    hardware = backend.get_hardware_name()

    if function not in dataset:
        return {
            "not_supported": True,
            "error": f"Function '{function}' not found in dataset",
            "error_type": "KeyError",
            "device": hardware
        }

    try:
        _prepare_backend_for_baseline(backend, language)
        ref_src = _load_reference_code_and_override_dims(function, batch_size, dim, input_dims)
        exec(ref_src, backend.context)
        result = _execute_baseline_with_error_handling(backend, function, language, hardware)

        if batch_size is not None:
            result['batch_size'] = batch_size
        if dim is not None:
            result['dim'] = dim
        if input_dims:
            result['input_dims'] = input_dims

        return result

    except Exception as e:
        print(f"[ERROR] Baseline computation failed: {e}")
        traceback.print_exc()
        return {
            "not_supported": True,
            "error": str(e),
            "error_type": type(e).__name__,
            "device": hardware
        }

    finally:
        try:
            backend.cleanup()
        except Exception:
            pass


def _override_num_perf_trials(language: str, num_trials: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    original_config = config.num_perf_trials
    config.num_perf_trials = num_trials

    original_performance = None
    if hasattr(performance_module, 'num_perf_trials'):
        original_performance = performance_module.num_perf_trials
        performance_module.num_perf_trials = num_trials

    original_pallas = None
    if language == 'pallas':
        try:
            import backends.pallas_backend as pallas_module
            if hasattr(pallas_module, 'num_perf_trials'):
                original_pallas = pallas_module.num_perf_trials
                pallas_module.num_perf_trials = num_trials
        except (ImportError, AttributeError):
            pass

    return original_config, original_performance, original_pallas


def _restore_num_perf_trials(language: str, original_config: Optional[int], original_performance: Optional[int], original_pallas: Optional[int]):
    if original_config is not None:
        config.num_perf_trials = original_config

    if original_performance is not None:
        performance_module.num_perf_trials = original_performance

    if original_pallas is not None and language == 'pallas':
        try:
            import backends.pallas_backend as pallas_module
            if hasattr(pallas_module, 'num_perf_trials'):
                pallas_module.num_perf_trials = original_pallas
        except (ImportError, AttributeError):
            pass


def _setup_performance_measurement_timeout(baseline_mean_ms: Optional[float], num_trials: int) -> Tuple[Optional[float], bool]:
    timeout_seconds = None
    use_timeout = False
    if baseline_mean_ms is not None and hasattr(signal, 'SIGALRM') and hasattr(signal, 'alarm'):
        total_baseline_time_ms = num_trials * baseline_mean_ms
        timeout_seconds = max(5.0, (5 * total_baseline_time_ms) / 1000.0)
        print(f"[INFO] Setting timeout for performance measurement: {timeout_seconds:.2f} seconds (5 * {num_trials} * baseline_mean = {total_baseline_time_ms:.2f}ms)")
        use_timeout = True
    elif baseline_mean_ms is not None:
        print(f"[WARNING] Timeout not available on this platform (signal.SIGALRM not supported)")

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
            print(f"[WARNING] Performance measurement returned None")
            return None, None
        if not isinstance(elapsed_times, (list, tuple, np.ndarray)):
            print(f"[WARNING] Performance measurement returned unexpected type")
            return None, None

        return elapsed_times, None

    except (TimeoutError, Exception) as e:
        print(f"[ERROR] Performance measurement failed: {e}")
        traceback.print_exc()
        return None, str(e)

    finally:
        if use_timeout and timeout_seconds is not None:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


def compute_all_baselines(language: str) -> Dict[str, Any]:
    backend = get_backend(language)
    hardware = backend.get_hardware_name()
    
    result = {}
    op_tests = dataset.keys()
    
    for op in op_tests:
        print(f'[INFO] Computing baseline for {op}')
        try:
            ref_src_path = os.path.join(str(REFERENCE_DIR), dataset[op]['category'], f'{op}.py')
            if not os.path.exists(ref_src_path):
                raise FileNotFoundError(f"Reference file not found: {ref_src_path}")

            with open(ref_src_path, 'r') as f:
                ref_src = f.read()

            exec(ref_src, backend.context)
            elapsed_times = backend.time_execution('Model')

            result[op] = {
                "mean": float(f"{np.mean(elapsed_times):.3g}"),
                "std": float(f"{np.std(elapsed_times):.3g}"),
                "min": float(f"{np.min(elapsed_times):.3g}"),
                "max": float(f"{np.max(elapsed_times):.3g}"),
                "num_trials": len(elapsed_times),
                'device': hardware
            }

        except Exception as e:
            print(f"[ERROR] {op}: {e}")
            result[op] = {
                "not_supported": True,
                "error": str(e),
                "error_type": type(e).__name__,
                "device": hardware
            }

        if language == 'cuda':
            _cleanup_cuda()

    return result
