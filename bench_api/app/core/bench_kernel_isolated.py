import subprocess
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# bench_api/ — the directory where `app/` lives, used as cwd for the worker
_WORKER_CWD = Path(__file__).parent.parent.parent


def _run_subprocess(params: Dict[str, Any], timeout: int, device_id: int = 0) -> Dict[str, Any]:
    child_env = os.environ.copy()
    child_env['CUDA_VISIBLE_DEVICES'] = str(device_id)

    try:
        proc = subprocess.run(
            [sys.executable, '-m', 'app.core.eval_worker'],
            input=json.dumps(params),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(_WORKER_CWD),
            env=child_env,
        )

        if proc.stdout.strip():
            return json.loads(proc.stdout.strip())

        error_msg = proc.stderr.strip() or 'No output from worker'
        return {
            'compiled': False,
            'correctness': None,
            'performance': None,
            'hardware': 'unknown',
            'error': f"Worker produced no output (exit={proc.returncode}): {error_msg[:1000]}",
        }

    except subprocess.TimeoutExpired:
        return {
            'compiled': False,
            'correctness': None,
            'performance': None,
            'hardware': 'unknown',
            'error': f"Evaluation timed out after {timeout}s",
        }
    except Exception as e:
        return {
            'compiled': False,
            'correctness': None,
            'performance': None,
            'hardware': 'unknown',
            'error': f"Failed to run worker: {e}",
        }


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

    # Subprocess 1: baseline + compile + correctness (CUDA_LAUNCH_BLOCKING=1 for accurate tracebacks)
    result = _run_subprocess({**base_params, 'mode': 'validation'}, timeout, device_id)

    if not result.get('compiled'):
        return result

    baseline_mean_ms = result.pop('baseline_mean_ms', None)

    # Subprocess 2: performance only, no CUDA_LAUNCH_BLOCKING
    # Runs regardless of correctness — speedup is useful even for incorrect kernels
    # Compilation cache from subprocess 1 is reused (torch_extensions / Triton cache on disk)
    perf_result = _run_subprocess(
        {**base_params, 'mode': 'benchmark', 'baseline_mean_ms': baseline_mean_ms},
        timeout,
        device_id,
    )

    result['performance'] = perf_result.get('performance')
    if perf_result.get('performance_error'):
        result['performance_error'] = perf_result['performance_error']
    result['timing']['performance'] = perf_result.get('timing', {}).get('performance', 0.0)
    result['timing']['total'] += perf_result.get('timing', {}).get('total', 0.0)

    return result
