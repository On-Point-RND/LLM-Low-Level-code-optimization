"""
Isolated evaluation worker.

Invoked as:  python -m app.core.eval_worker   (cwd = bench_api/)
Reads JSON params from stdin, writes a single JSON result line to stdout.
stderr is for logs only.
"""

import json
import os
import signal
import sys
from dataclasses import asdict, is_dataclass


def _write(result) -> None:
    if is_dataclass(result):
        result = asdict(result)

    fd = int(os.environ.get("RESULT_FD", -1))
    data = json.dumps(result).encode()
    if fd >= 0:
        os.write(fd, data)
        os.close(fd)
    else:
        sys.stdout.buffer.write(data + b"\n")
        sys.stdout.flush()


def _signal_handler(signum, frame):
    try:
        from app.core.phases.common import get_current_eval_stage

        stage = get_current_eval_stage()
    except Exception:
        stage = "initialization"

    sig_name = signal.Signals(signum).name
    error_msg = f"Process received signal {signum} ({sig_name}) during {stage} stage"
    result = {
        "compiled": stage not in ["initialization", "baseline", "compilation"],
        "correctness": False if stage == "correctness" else None,
        "performance": None,
        "hardware": "unknown",
        "stage": stage,
    }
    if stage in ("initialization", "baseline"):
        result["system_error"] = error_msg
    elif stage == "compilation":
        result["compile_info"] = error_msg
    elif stage == "correctness":
        result["correctness_info"] = error_msg
    elif stage == "performance":
        result["performance_info"] = error_msg
    else:
        result["system_error"] = error_msg

    _write(result)
    os._exit(signum)


def main():
    for sig in [signal.SIGSEGV, signal.SIGFPE, signal.SIGABRT, signal.SIGBUS]:
        signal.signal(sig, _signal_handler)

    params = json.loads(sys.stdin.read())
    mode = params.get("mode", "validation")

    if mode != "benchmark":
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    from app.integration import register_kernelbench_dataset

    register_kernelbench_dataset()

    from app.core.backends.backend_registry import get_backend

    backend = None

    try:
        backend = get_backend(params["language"])
        if backend is None:
            _write(
                {
                    "compiled": False,
                    "correctness": None,
                    "performance": None,
                    "hardware": "unknown",
                    "system_error": f"Backend {params['language']} not found",
                }
            )
            return

        hardware = backend.get_hardware_name()
        capability = backend.get_compute_capability()
        compute_capability = f"{capability[0]}.{capability[1]}" if capability else None

        if mode == "compilation":
            from app.core.phases.compilation import run_compilation_phase

            result = run_compilation_phase(
                backend,
                params["function_code"],
                params["function"],
                hardware,
                compute_capability,
            )
        elif mode == "baseline":
            from app.core.phases.baseline import run_baseline_phase

            result = run_baseline_phase(
                backend,
                params["function"],
                hardware,
                compute_capability,
                batch_size=params.get("batch_size"),
                dim=params.get("dim"),
                input_dims=params.get("input_dims"),
                torch_compile=params.get("torch_compile_baseline", False),
                num_trials=params.get("num_trials"),
                num_warmup=params.get("num_warmup"),
            )
        elif mode == "validation":
            from app.core.phases.validation import run_validation_phase

            result = run_validation_phase(
                backend,
                params["function_code"],
                params["function"],
                hardware,
                compute_capability,
                batch_size=params.get("batch_size"),
                dim=params.get("dim"),
                input_dims=params.get("input_dims"),
            )
        elif mode == "benchmark":
            from app.core.phases.benchmark import run_benchmark_phase

            result = run_benchmark_phase(
                backend,
                params["function_code"],
                params["function"],
                hardware,
                compute_capability,
                batch_size=params.get("batch_size"),
                dim=params.get("dim"),
                input_dims=params.get("input_dims"),
                torch_compile=params.get("torch_compile", False),
                num_trials=params.get("num_trials"),
                num_warmup=params.get("num_warmup"),
                baseline_mean_ms=params.get("baseline_mean_ms"),
            )
        else:
            result = {
                "compiled": False,
                "correctness": None,
                "performance": None,
                "hardware": hardware,
                "system_error": f"Unknown mode: {mode}",
            }

    except Exception as e:
        import traceback

        result = {
            "compiled": False,
            "correctness": None,
            "performance": None,
            "hardware": "unknown",
            "system_error": f"Evaluation failed: {e}\n{traceback.format_exc()}",
        }
    finally:
        try:
            if backend:
                backend.cleanup()
        except Exception:
            pass

    _write(result)


if __name__ == "__main__":
    main()
