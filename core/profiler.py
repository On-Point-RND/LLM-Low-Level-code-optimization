import numpy as np
import tvm
from tvm import relax

def profile_relax(mod, inputs_tvm, dev, target="cuda", func_name="main", warmup_iters=10, number=50, repeat=20):
    target_obj = tvm.target.Target(target)
    ex = relax.build(mod, target_obj)
    vm = relax.VirtualMachine(ex, dev, profile=True)
    
    for _ in range(warmup_iters):
        _ = vm[func_name](*inputs_tvm)
    
    dev.sync()
    
    ftimer = vm.time_evaluator(func_name, dev, number=number, repeat=repeat)
    prof = ftimer(*inputs_tvm)
    results_ms = np.array(prof.results) * 1e3
    
    mean_ms = float(np.mean(results_ms))
    std_ms = float(np.std(results_ms))
    min_ms = float(np.min(results_ms))
    max_ms = float(np.max(results_ms))
    median_ms = float(np.median(results_ms))
    p95_ms = float(np.percentile(results_ms, 95))
    p99_ms = float(np.percentile(results_ms, 99))
    
    prof_report = vm.profile(func_name, *inputs_tvm)
    
    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "profiler_report": prof_report
    }

def profile_tir(mod, inputs_tvm, dev, target="cuda", func_name="main", warmup_iters=10, number=50, repeat=20):
    target_obj = tvm.target.Target(target)
    ex = relax.build(mod, target_obj)
    vm = relax.VirtualMachine(ex, dev, profile=True)
    
    for _ in range(warmup_iters):
        _ = vm[func_name](*inputs_tvm)
    
    dev.sync()
    
    ftimer = vm.time_evaluator(func_name, dev, number=number, repeat=repeat)
    prof = ftimer(*inputs_tvm)
    results_ms = np.array(prof.results) * 1e3
    
    mean_ms = float(np.mean(results_ms))
    std_ms = float(np.std(results_ms))
    min_ms = float(np.min(results_ms))
    max_ms = float(np.max(results_ms))
    median_ms = float(np.median(results_ms))
    p95_ms = float(np.percentile(results_ms, 95))
    p99_ms = float(np.percentile(results_ms, 99))
    
    prof_report = vm.profile(func_name, *inputs_tvm)
    
    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "profiler_report": prof_report
    }
