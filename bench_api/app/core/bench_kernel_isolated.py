import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

ISOLATED_SCRIPT = Path(__file__).parent.parent.parent / 'eval_single_isolated.py'


def _run_subprocess(params: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    """Run eval_single_isolated.py with the given params, return parsed JSON result."""
    try:
        result = subprocess.run(
            [sys.executable, str(ISOLATED_SCRIPT)],
            input=json.dumps(params),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ISOLATED_SCRIPT.parent.parent)
        )

        stdout_text = result.stdout.strip() if result.stdout else ""
        stderr_text = result.stderr.strip() if result.stderr else ""

        json_output = None
        for text in (stderr_text, stdout_text):
            if not text:
                continue
            for line in reversed(text.split('\n')):
                line = line.strip()
                if line and (line.startswith('{') or line.startswith('[')):
                    try:
                        json_output = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            if json_output is not None:
                break

        if json_output is not None:
            return json_output

        error_msg = stderr_text or stdout_text or "Unknown error"
        return {
            'compiled': False,
            'correctness': None,
            'performance': None,
            'hardware': 'unknown',
            'compile_info': f"Subprocess failed (returncode={result.returncode}): {error_msg[:1000]}",
            'error': f"Failed to find JSON in subprocess output (returncode={result.returncode}): {error_msg[:1000]}"
        }

    except subprocess.TimeoutExpired:
        return {
            'compiled': False,
            'correctness': None,
            'performance': None,
            'hardware': 'unknown',
            'compile_info': f"Evaluation timed out after {timeout} seconds",
            'error': f"Evaluation timed out after {timeout} seconds"
        }
    except Exception as e:
        return {
            'compiled': False,
            'correctness': None,
            'performance': None,
            'hardware': 'unknown',
            'compile_info': f"Subprocess execution failed: {str(e)}",
            'error': f"Subprocess execution failed: {str(e)}"
        }


def evaluate_kernel_isolated(
    function_code: str,
    function: str,
    language: str,
    torch_compile: bool = False,
    num_trials: Optional[int] = None,
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None,
    timeout: int = 300
) -> Dict[str, Any]:
    base_params = {
        'function_code': function_code,
        'function': function,
        'language': language,
        'torch_compile': torch_compile,
        'num_trials': num_trials,
        'batch_size': batch_size,
        'dim': dim,
        'input_dims': input_dims,
    }

    # Subprocess 1: baseline + compile + correctness (CUDA_LAUNCH_BLOCKING=1 for accurate tracebacks)
    result = _run_subprocess({**base_params, 'mode': 'validation'}, timeout)

    if not result.get('compiled'):
        return result

    baseline_mean_ms = result.pop('baseline_mean_ms', None)

    # Subprocess 2: performance only, no CUDA_LAUNCH_BLOCKING
    # Runs regardless of correctness — speedup is useful even for incorrect kernels
    perf_result = _run_subprocess(
        {**base_params, 'mode': 'benchmark', 'baseline_mean_ms': baseline_mean_ms},
        timeout
    )

    result['performance'] = perf_result.get('performance')
    if perf_result.get('performance_error'):
        result['performance_error'] = perf_result['performance_error']
    result['timing']['performance'] = perf_result.get('timing', {}).get('performance', 0.0)
    result['timing']['total'] += perf_result.get('timing', {}).get('total', 0.0)

    return result
