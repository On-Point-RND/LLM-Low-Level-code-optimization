"""
CUDA kernel profiling via torch.profiler.

Runs a small number of profiled iterations to capture per-kernel GPU time,
call counts, and estimated FLOPs. Results are suitable for feeding back into
the LLM prompt to guide the next optimization attempt.
"""
import logging
from typing import Optional

import torch
from torch.profiler import ProfilerActivity, profile

from app.core.phases.common import InfraError

logger = logging.getLogger(__name__)

# Number of profiled iterations (fewer than timing trials — profiler has overhead).
_PROFILE_RUNS = 5
_PROFILE_WARMUP = 2
# Maximum number of top kernels to return.
_TOP_K = 10


def profile_kernel(
    backend,
    eval_target: str = "ModelNew",
    num_runs: int = _PROFILE_RUNS,
    warmup: int = _PROFILE_WARMUP,
) -> Optional[list]:
    """
    Run torch.profiler on eval_target and return a sorted breakdown of
    the top CUDA kernels by self GPU time.

    Returns a list of dicts:
        {"name": str, "self_cuda_us": float, "count": int, "flops": int|None}
    sorted descending by self_cuda_us.  Returns None on any failure.

    Raises InfraError if infrastructure (input preparation) fails.
    """
    context = backend.context
    device = backend.get_device()

    try:
        init_inputs = [
            x.to(device) if isinstance(x, torch.Tensor) else x
            for x in context["get_init_inputs"]()
        ]
    except Exception as e:
        raise InfraError(f"profile_kernel: failed to prepare init inputs: {e}") from e

    with torch.no_grad():
        model = context[eval_target](*init_inputs).to(device)
        backend.synchronize()

        # Warmup outside the profiler to ensure kernel JIT is done
        for _ in range(warmup):
            try:
                inputs = [
                    x.to(device) if isinstance(x, torch.Tensor) else x
                    for x in context["get_inputs"]()
                ]
            except Exception as e:
                raise InfraError(f"profile_kernel: failed to prepare inputs: {e}") from e
            model(*inputs)
        backend.synchronize()

        # Profiled runs
        with profile(
            activities=[ProfilerActivity.CUDA],
            record_shapes=False,
            with_flops=True,
        ) as prof:
            for _ in range(num_runs):
                try:
                    inputs = [
                        x.to(device) if isinstance(x, torch.Tensor) else x
                        for x in context["get_inputs"]()
                    ]
                except Exception as e:
                    raise InfraError(f"profile_kernel: failed to prepare inputs: {e}") from e
                model(*inputs)
            backend.synchronize()

    events = prof.key_averages()
    if not events:
        return []

    results = []
    for e in sorted(events, key=lambda x: x.self_cuda_time_total, reverse=True)[:_TOP_K]:
        results.append({
            "name": e.key,
            "self_cuda_us": e.self_cuda_time_total,
            "count": e.count,
            "flops": getattr(e, "flops", None) or None,
        })

    return results
