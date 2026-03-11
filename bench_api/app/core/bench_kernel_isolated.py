import subprocess
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# bench_api/ — the directory where `app/` lives, used as cwd for the worker
_WORKER_CWD = Path(__file__).parent.parent.parent


def _error(msg: str) -> Dict[str, Any]:
    return {
        'compiled': False,
        'correctness': None,
        'performance': None,
        'hardware': 'unknown',
        'error': msg,
        'system_error': True,
    }


def _run_subprocess(params: Dict[str, Any], timeout: int, device_id: int = 0) -> Dict[str, Any]:
    r_fd, w_fd = os.pipe()

    child_env = os.environ.copy()
    child_env['CUDA_VISIBLE_DEVICES'] = str(device_id)
    child_env['RESULT_FD'] = str(w_fd)

    try:
        proc = subprocess.Popen(
            [sys.executable, '-m', 'app.core.eval_worker'],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd=str(_WORKER_CWD),
            env=child_env,
            pass_fds=(w_fd,),
        )
        os.close(w_fd)

        proc.stdin.write(json.dumps(params).encode())
        proc.stdin.close()

        with os.fdopen(r_fd, 'r') as result_pipe:
            result_json = result_pipe.read()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return _error(f"Evaluation timed out after {timeout}s")

        if result_json:
            return json.loads(result_json)

        stderr = proc.stderr.read().strip() if proc.stderr else ''
        return _error(f"Worker produced no output (exit={proc.returncode}): {stderr[:1000]}")

    except Exception as e:
        try:
            os.close(r_fd)
        except OSError:
            pass
        try:
            os.close(w_fd)
        except OSError:
            pass
        return _error(f"Failed to run worker: {e}")


def evaluate_kernel_isolated(
    function_code: str,
    function: str,
    language: str,
    torch_compile: bool = False,
    torch_compile_baseline: bool = False,
    num_trials: Optional[int] = None,
    num_warmup: Optional[int] = None,
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None,
    timeout: int = 300,
    device_id: int = 0,
    mode: str = 'full',
    baseline_mean_ms: Optional[float] = None,
) -> Dict[str, Any]:
    base_params = {
        'function_code': function_code,
        'function': function,
        'language': language,
        'torch_compile': torch_compile,
        'torch_compile_baseline': torch_compile_baseline,
        'num_trials': num_trials,
        'num_warmup': num_warmup,
        'batch_size': batch_size,
        'dim': dim,
        'input_dims': input_dims,
    }

    result = {}

    # Phase A: Compilation (Subprocess A)
    if mode in ('compilation', 'full'):
        result = _run_subprocess({**base_params, 'mode': 'compilation'}, timeout, device_id)
        if mode == 'compilation' or not result.get('compiled'):
            return result

    # Phase B: Validation (Subprocess B)
    if mode in ('validation_benchmark', 'full'):
        # Subprocess B: baseline + compile (cache) + correctness
        # Use baseline_mean_ms if provided (e.g. from a previous call)
        val_params = {**base_params, 'mode': 'validation'}
        if baseline_mean_ms is not None:
            val_params['baseline_mean_ms'] = baseline_mean_ms
            
        val_result = _run_subprocess(val_params, timeout, device_id)
        
        # Merge results. If we're in 'full' mode, 'result' already has compilation info.
        # If we're in 'validation_benchmark', 'result' is empty.
        if not result:
            result = val_result
        else:
            # Update result with validation info
            result.update({
                'correctness': val_result.get('correctness'),
                'correctness_info': val_result.get('correctness_info'),
                'error': val_result.get('error'),
                'stage': val_result.get('stage'),
            })
            if 'timing' in val_result:
                result['timing']['correctness'] = val_result['timing'].get('correctness', 0.0)
                result['timing']['total'] += val_result['timing'].get('total', 0.0)

        if not val_result.get('compiled'):
            return result
            
        current_baseline_mean_ms = val_result.get('baseline_mean_ms') or baseline_mean_ms

        # Phase C: Benchmark (Subprocess C)
        perf_result = _run_subprocess(
            {**base_params, 'mode': 'benchmark', 'baseline_mean_ms': current_baseline_mean_ms},
            timeout,
            device_id,
        )

        result['performance'] = perf_result.get('performance')
        if perf_result.get('performance_error'):
            result['performance_error'] = perf_result['performance_error']
            if not result.get('error'):
                result['error'] = perf_result['performance_error']
        
        if 'timing' in perf_result:
            if 'timing' not in result:
                result['timing'] = perf_result['timing']
            else:
                result['timing']['performance'] = perf_result['timing'].get('performance', 0.0)
                result['timing']['total'] += perf_result['timing'].get('total', 0.0)
        
        # Preserve baseline_mean_ms for the caller if needed
        if current_baseline_mean_ms:
            result['baseline_mean_ms'] = current_baseline_mean_ms

    return result
