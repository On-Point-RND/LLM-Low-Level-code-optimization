import torch
import app.config as config
from typing import Optional


def run_performance(backend, eval_target: str = 'ModelNew', num_trials: Optional[int] = None, torch_compile: bool = False):
    """
    Run warmup + timed trials for eval_target using backend timing primitives.
    Returns list of elapsed times in ms.
    """
    actual_trials = num_trials if num_trials is not None else config.NUM_PERF_TRIALS

    context = backend.context
    device = backend.get_device()

    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in context['get_inputs']()]
    init_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in context['get_init_inputs']()]

    with torch.no_grad():
        model = context[eval_target](*init_inputs).to(device)

        if torch_compile:
            try:
                model = torch.compile(model)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"torch.compile failed, using original model: {e}")

        for _ in range(config.NUM_WARMUP):
            model(*inputs)
            backend.synchronize()

        return [backend.elapsed_ms(lambda: model(*inputs)) for _ in range(actual_trials)]
