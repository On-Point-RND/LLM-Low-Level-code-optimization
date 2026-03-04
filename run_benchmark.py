#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
tvm_python_dir = script_dir / "tvm" / "python"
if str(tvm_python_dir) not in sys.path:
    sys.path.insert(0, str(tvm_python_dir))


def find_baseline_files(kernelbench_root):
    baseline_files = []
    root = Path(kernelbench_root)
    for py_file in root.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        if ".ipynb_checkpoints" in str(py_file):
            continue
        baseline_files.append(py_file)
    return sorted(baseline_files)

def _run_single_baseline(baseline_file, config_path, kernelbench_root, res_root, gpu_id):
    """Run process_baseline for one file (used in subprocess)."""
    from core.pipeline import process_baseline
    return process_baseline(
        str(baseline_file), str(config_path), str(kernelbench_root), str(res_root), gpu_id=gpu_id
    )


def _run_in_subprocess(baseline_file, gpu_id, log_file=None, timeout=5000):
    """Spawn subprocess for one baseline; stream output in real-time. Returns exit code (-99 on timeout)."""
    import selectors
    import time as _time

    real_stdout = sys.__stdout__
    real_stderr = sys.__stderr__

    proc = subprocess.Popen(
        [
            sys.executable, "-u",
            str(Path(__file__).resolve()),
            "--run-single", str(baseline_file.resolve()),
            "--gpu", str(gpu_id),
        ],
        cwd=str(script_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    sel.register(proc.stderr, selectors.EVENT_READ)

    deadline = _time.monotonic() + timeout
    open_streams = 2
    while open_streams > 0:
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            proc.kill()
            proc.wait()
            sel.close()
            return -99
        for key, _ in sel.select(timeout=min(remaining, 5.0)):
            chunk = key.fileobj.read1(8192) if hasattr(key.fileobj, "read1") else key.fileobj.read(8192)
            if not chunk:
                sel.unregister(key.fileobj)
                open_streams -= 1
                continue
            text = chunk.decode("utf-8", errors="replace")
            target = real_stdout if key.fileobj is proc.stdout else real_stderr
            target.write(text)
            target.flush()
            if log_file:
                log_file.write(text)
                log_file.flush()

    sel.close()
    proc.wait()
    return proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--skip-errors", action="store_true", help="Skip failed tasks instead of crashing")
    parser.add_argument("--run-single", type=str, metavar="BASELINE_FILE", help="Run one baseline in subprocess (internal)")
    args = parser.parse_args()

    kernelbench_root = script_dir / "KernelBench"
    res_root = script_dir / "res"
    config_path = script_dir / "config.yaml"

    if args.run_single:
        baseline_file = Path(args.run_single)
        _run_single_baseline(baseline_file, config_path, kernelbench_root, res_root, args.gpu)
        return

    os.makedirs(res_root, exist_ok=True)
    output_log_path = res_root / "output.log"

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data: str) -> None:
            for s in self._streams:
                s.write(data)
                s.flush()

        def flush(self) -> None:
            for s in self._streams:
                s.flush()

    log_file = open(output_log_path, "a", encoding="utf-8")
    _log_file = log_file
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    if not config_path.exists():
        print(f"Error: config.yaml not found at {config_path}")
        sys.exit(1)

    baseline_files = find_baseline_files(kernelbench_root)
    if not baseline_files:
        print("No baseline files found in KernelBench/")
        sys.exit(1)

    print("=" * 70)
    print("TVM Compiler Benchmark")
    print("=" * 70)
    print(f"Found {len(baseline_files)} baseline file(s)")
    print(f"GPU: {args.gpu}")
    print(f"Config: {config_path}")
    print(f"Results directory: {res_root}")
    print("=" * 70)
    print()

    results = []
    error_list = []
    start_time = __import__("time").time()

    for i, baseline_file in enumerate(baseline_files, 1):
        model_path_rel = baseline_file.relative_to(kernelbench_root)
        task_name = str(model_path_rel).replace(".py", "")
        metrics_path = res_root / task_name / "metrics.json"

        print("=" * 70)
        print(f"[{i}/{len(baseline_files)}] Processing: {model_path_rel}")
        print("=" * 70)

        if metrics_path.exists():
            with open(metrics_path) as _f:
                metrics = json.load(_f)
            results.append(metrics)
            print("  [SKIP] Already processed, metrics.json exists")
            print()
            continue

        returncode = _run_in_subprocess(baseline_file, args.gpu, log_file=_log_file)

        if returncode != 0:
            err_msg = f"exit code {returncode}" if returncode != -99 else "timeout (600s)"
            error_list.append({"task": str(model_path_rel), "error": err_msg})
            print(f"  [ERROR] Subprocess failed: {err_msg}")
            print()
            continue

        if metrics_path.exists():
            with open(metrics_path) as _f:
                metrics = json.load(_f)
            results.append(metrics)
            print()
            print("  Summary:")
            print(f"    Model: {metrics.get('name', 'unknown')}")
            if "latency" in metrics:
                if "after_tir" in metrics["latency"]:
                    at = metrics["latency"]["after_tir"]
                    if at.get("skipped"):
                        print(f"    Latency (after TIR): skipped ({at.get('reason', 'llm_failed')})")
                    else:
                        print(f"    Latency (after TIR): {at['mean_ms']:.3f} ms")
                elif "after_relax" in metrics["latency"]:
                    print(f"    Latency (after Relax): {metrics['latency']['after_relax']['mean_ms']:.3f} ms")
            if "correctness" in metrics:
                status = "PASS" if metrics["correctness"]["is_correct"] else "FAIL"
                print(f"    Correctness: {status}")
            print(f"    Results saved to: {res_root / task_name}")
        print()
    
    elapsed_time = __import__('time').time() - start_time
    print("=" * 70)
    print("Benchmark Complete")
    print("=" * 70)
    print(f"Processed: {len(results)}/{len(baseline_files)} baselines successfully")
    if error_list:
        print(f"Errors: {len(error_list)}")
        for err in error_list:
            print(f"  - {err['task']}: {err['error']}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per model: {elapsed_time/len(baseline_files):.2f} seconds")
    print()
    
    summary_file = res_root / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    if error_list:
        errors_file = res_root / "errors.json"
        with open(errors_file, "w") as f:
            json.dump(error_list, f, indent=2)
        print(f"Errors saved to: {errors_file}")

    print("=" * 70)

if __name__ == "__main__":
    main()
