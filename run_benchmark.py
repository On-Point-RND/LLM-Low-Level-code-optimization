#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
tvm_python_dir = script_dir / "tvm" / "python"
if str(tvm_python_dir) not in sys.path:
    sys.path.insert(0, str(tvm_python_dir))

from core.pipeline import process_baseline

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--skip-errors", action="store_true", help="Skip failed tasks instead of crashing")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.absolute()
    kernelbench_root = script_dir / "KernelBench"
    res_root = script_dir / "res"
    config_path = script_dir / "config.yaml"

    # Глобальный лог всего вывода (stdout/stderr) в один файл
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
    
    import json
    import traceback

    results = []
    error_list = []
    start_time = __import__('time').time()
    
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

        try:
            metrics = process_baseline(
                str(baseline_file),
                str(config_path),
                str(kernelbench_root),
                str(res_root),
                gpu_id=args.gpu,
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  [ERROR] {e.__class__.__name__}: {e}")
            print(tb)
            error_list.append({"task": str(model_path_rel), "error": str(e), "traceback": tb})
            print()
            if args.skip_errors:
                continue
            raise

        results.append(metrics)
        print()
        print("  Summary:")
        print(f"    Model: {metrics.get('name', 'unknown')}")
        if "latency" in metrics:
            if "after_tir" in metrics["latency"]:
                print(f"    Latency (after TIR): {metrics['latency']['after_tir']['mean_ms']:.3f} ms")
            elif "after_relax" in metrics["latency"]:
                print(f"    Latency (after Relax): {metrics['latency']['after_relax']['mean_ms']:.3f} ms")
        if "correctness" in metrics:
            status = "✓ PASS" if metrics["correctness"]["is_correct"] else "✗ FAIL"
            print(f"    Correctness: {status}")
        print(f"    Results saved to: {res_root / model_path_rel}")
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
