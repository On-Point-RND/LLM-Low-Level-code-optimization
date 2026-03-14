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
    child_env.pop('CUDA_LAUNCH_BLOCKING', None)
    child_env.pop('TORCH_USE_CUDA_DSA', None)

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


def run_baseline(params: Dict[str, Any], timeout: int = 300, device_id: int = 0) -> Dict[str, Any]:
    return _run_subprocess({**params, 'mode': 'baseline'}, timeout, device_id)


def run_compilation(params: Dict[str, Any], timeout: int = 300, device_id: int = 0) -> Dict[str, Any]:
    return _run_subprocess({**params, 'mode': 'compilation'}, timeout, device_id)


def run_validation(params: Dict[str, Any], timeout: int = 300, device_id: int = 0, baseline_mean_ms: Optional[float] = None) -> Dict[str, Any]:
    p = {**params, 'mode': 'validation'}
    if baseline_mean_ms is not None:
        p['baseline_mean_ms'] = baseline_mean_ms
    return _run_subprocess(p, timeout, device_id)


def run_benchmark(params: Dict[str, Any], timeout: int = 300, device_id: int = 0, baseline_mean_ms: Optional[float] = None) -> Dict[str, Any]:
    p = {**params, 'mode': 'benchmark'}
    if baseline_mean_ms is not None:
        p['baseline_mean_ms'] = baseline_mean_ms
    return _run_subprocess(p, timeout, device_id)
