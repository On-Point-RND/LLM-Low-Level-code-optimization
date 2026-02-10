#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from core.pipeline import process_baseline

def find_baseline_files(dataset_root):
    baseline_files = []
    dataset_path = Path(dataset_root)
    
    for py_file in dataset_path.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        if ".ipynb_checkpoints" in str(py_file):
            continue
        baseline_files.append(py_file)
    
    return sorted(baseline_files)

def main():
    script_dir = Path(__file__).parent.absolute()
    dataset_root = script_dir / "dataset"
    res_root = script_dir / "res"
    config_path = dataset_root / "config.yaml"
    
    if not config_path.exists():
        print(f"Error: config.yaml not found at {config_path}")
        sys.exit(1)
    
    baseline_files = find_baseline_files(dataset_root)
    
    if not baseline_files:
        print("No baseline files found in dataset/")
        sys.exit(1)
    
    print("=" * 70)
    print("TVM Compiler Benchmark")
    print("=" * 70)
    print(f"Found {len(baseline_files)} baseline file(s)")
    print(f"Config: {config_path}")
    print(f"Results directory: {res_root}")
    print("=" * 70)
    print()
    
    results = []
    start_time = __import__('time').time()
    
    for i, baseline_file in enumerate(baseline_files, 1):
        model_path_rel = baseline_file.relative_to(dataset_root)
        print("=" * 70)
        print(f"[{i}/{len(baseline_files)}] Processing: {model_path_rel}")
        print("=" * 70)
        
        try:
            metrics = process_baseline(
                str(baseline_file),
                str(config_path),
                str(dataset_root),
                str(res_root)
            )
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
        except Exception as e:
            print()
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    elapsed_time = __import__('time').time() - start_time
    print("=" * 70)
    print("Benchmark Complete")
    print("=" * 70)
    print(f"Processed: {len(results)}/{len(baseline_files)} baselines successfully")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per model: {elapsed_time/len(baseline_files):.2f} seconds")
    print()
    
    summary_file = res_root / "summary.json"
    import json
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to: {summary_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
