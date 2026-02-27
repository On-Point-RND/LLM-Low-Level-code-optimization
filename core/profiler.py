import numpy as np
import tvm
from tvm import relax


def _stats_from_results(results_ms):
    return {
        "mean_ms": float(np.mean(results_ms)),
        "std_ms": float(np.std(results_ms)),
        "min_ms": float(np.min(results_ms)),
        "max_ms": float(np.max(results_ms)),
        "median_ms": float(np.median(results_ms)),
        "p95_ms": float(np.percentile(results_ms, 95)),
        "p99_ms": float(np.percentile(results_ms, 99)),
    }


def _run_benchmark(vm, func_name, inputs_tvm, dev, warmup_iters, number, repeat):
    for _ in range(warmup_iters):
        _ = vm[func_name](*inputs_tvm)
    dev.sync()
    ftimer = vm.time_evaluator(func_name, dev, number=number, repeat=repeat)
    prof = ftimer(*inputs_tvm)
    return np.array(prof.results) * 1e3


def profile_tir(mod, inputs_tvm, dev, target="cuda", func_name="main", warmup_iters=10, number=50, repeat=20):
    target_obj = tvm.target.Target(target)
    ex = relax.build(mod, target_obj)
    vm = relax.VirtualMachine(ex, dev, profile=False)
    results_ms = _run_benchmark(vm, func_name, inputs_tvm, dev, warmup_iters, number, repeat)
    out = _stats_from_results(results_ms)
    vm_profile = relax.VirtualMachine(ex, dev, profile=True)
    out["profiler_report"] = vm_profile.profile(func_name, *inputs_tvm)
    return out


def profile_executable(ex, inputs_tvm, dev, func_name="main", warmup_iters=3, number=10, repeat=5, skip_prof_report=False, verbose=False):
    def log(s):
        if verbose:
            print(f"      {s}", flush=True)
    log("Creating VM...")
    vm = relax.VirtualMachine(ex, dev, profile=False)
    dev.sync()
    log(f"Warmup ({warmup_iters} iters)...")
    log(f"Benchmark (number={number}, repeat={repeat})...")
    results_ms = _run_benchmark(vm, func_name, inputs_tvm, dev, warmup_iters, number, repeat)
    out = _stats_from_results(results_ms)
    if skip_prof_report:
        out["profiler_report"] = None
    else:
        vm_profile = relax.VirtualMachine(ex, dev, profile=True)
        out["profiler_report"] = vm_profile.profile(func_name, *inputs_tvm)
    return out
