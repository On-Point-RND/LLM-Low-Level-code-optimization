import subprocess
import json
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Set

from app.core.backends.backend_registry import get_subprocess_env

# bench_api/ — the directory where `app/` lives, used as cwd for the worker
_WORKER_CWD = Path(__file__).parent.parent.parent

# Registry of all currently-running worker processes so they can be killed on shutdown.
_active_procs_lock = threading.Lock()
_active_procs: Set[subprocess.Popen] = set()


def shutdown_all_workers() -> None:
    """Kill every active worker subprocess. Called during server shutdown."""
    with _active_procs_lock:
        procs = list(_active_procs)
    for proc in procs:
        try:
            proc.kill()
        except Exception:
            pass
    for proc in procs:
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


def _error(msg: str) -> Dict[str, Any]:
    return {
        "compiled": False,
        "correctness": None,
        "performance": None,
        "hardware": "unknown",
        "system_error": msg,
    }


def _run_subprocess(
    params: Dict[str, Any], timeout: int, device_id: int = 0
) -> Dict[str, Any]:
    r_fd, w_fd = os.pipe()

    child_env = os.environ.copy()
    child_env["RESULT_FD"] = str(w_fd)
    benchmark_mode = params.get("mode") == "benchmark"
    for key, value in get_subprocess_env(
        params["language"], device_id, benchmark_mode, req_id=params.get("req_id")
    ).items():
        if value is None:
            child_env.pop(key, None)
        else:
            child_env[key] = value

    proc: Optional[subprocess.Popen] = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "app.core.eval_worker"],
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

        with _active_procs_lock:
            _active_procs.add(proc)

        # Read the result pipe in a daemon thread so that proc.wait(timeout=...)
        # can enforce the deadline even if the child never closes the write end
        # (e.g. hangs inside a CUDA kernel without dying).
        result_holder: Dict[str, str] = {}

        def _read_pipe() -> None:
            try:
                with os.fdopen(r_fd, "r") as result_pipe:
                    result_holder["data"] = result_pipe.read()
            except Exception:
                pass

        reader = threading.Thread(target=_read_pipe, daemon=True)
        reader.start()

        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            timed_out = True

        # Give the reader a short window to drain whatever the child wrote before dying.
        reader.join(timeout=5)

        if timed_out:
            return _error(f"Evaluation timed out after {timeout}s")

        result_json = result_holder.get("data", "")
        if result_json:
            return json.loads(result_json)

        stderr = proc.stderr.read().strip() if proc.stderr else ""
        return _error(
            f"Worker produced no output (exit={proc.returncode}): {stderr[-2000:]}"
        )

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

    finally:
        if proc is not None:
            with _active_procs_lock:
                _active_procs.discard(proc)



def run_baseline(
    params: Dict[str, Any], timeout: int = 3600, device_id: int = 0
) -> Dict[str, Any]:
    return _run_subprocess({**params, "mode": "baseline"}, timeout, device_id)


def run_compilation(
    params: Dict[str, Any], timeout: int = 3600, device_id: int = 0
) -> Dict[str, Any]:
    return _run_subprocess({**params, "mode": "compilation"}, timeout, device_id)


def run_validation(
    params: Dict[str, Any],
    timeout: int = 3600,
    device_id: int = 0,
    baseline_mean_ms: Optional[float] = None,
) -> Dict[str, Any]:
    p = {**params, "mode": "validation"}
    if baseline_mean_ms is not None:
        p["baseline_mean_ms"] = baseline_mean_ms
    return _run_subprocess(p, timeout, device_id)


def run_benchmark(
    params: Dict[str, Any],
    timeout: int = 3600,
    device_id: int = 0,
    baseline_mean_ms: Optional[float] = None,
) -> Dict[str, Any]:
    p = {**params, "mode": "benchmark"}
    if baseline_mean_ms is not None:
        p["baseline_mean_ms"] = baseline_mean_ms
    return _run_subprocess(p, timeout, device_id)
