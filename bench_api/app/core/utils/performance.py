import logging
import torch
import app.config as config
from typing import Optional

from app.core.phases.common import InfraError

logger = logging.getLogger(__name__)


def run_performance(
    backend,
    eval_target: str = "ModelNew",
    num_trials: Optional[int] = None,
    num_warmup: Optional[int] = None,
    torch_compile: bool = False,
):
    """
    Run warmup + timed trials for eval_target using backend timing primitives.
    Returns list of elapsed times in ms.
    Raises InfraError for infrastructure failures (our fault).
    Raises other exceptions for kernel execution failures (user fault).
    """
    actual_trials = num_trials if num_trials is not None else config.NUM_PERF_TRIALS
    actual_warmup = num_warmup if num_warmup is not None else config.NUM_WARMUP

    context = backend.context
    device = backend.get_device()

    # Infra: no user code has run yet, all failures are our fault
    try:
        init_inputs = [
            x.to(device) if isinstance(x, torch.Tensor) else x
            for x in context["get_init_inputs"]()
        ]
    except TimeoutError:
        raise
    except Exception as e:
        raise InfraError(f"Failed to prepare init inputs: {e}") from e

    with torch.no_grad():
        # User code: model instantiation, then sync immediately to flush deferred errors
        model = context[eval_target](*init_inputs).to(device)
        backend.synchronize()

        if torch_compile:
            try:
                model = torch.compile(model)
            except Exception as e:
                raise InfraError(f"torch.compile failed: {e}") from e

        torch.manual_seed(config.SEED_NUM)

        for _ in range(actual_warmup):
            # Infra: previous user code was fully synced, failures here are our fault
            try:
                inputs = [
                    x.to(device) if isinstance(x, torch.Tensor) else x
                    for x in context["get_inputs"]()
                ]
            except TimeoutError:
                raise
            except Exception as e:
                raise InfraError(f"Failed to prepare inputs: {e}") from e

            # User code: forward pass, then sync immediately
            model(*inputs)
            backend.synchronize()

        elapsed = []
        for _ in range(actual_trials):
            # Infra: previous user code was fully synced (via elapsed_ms), our fault
            try:
                inputs = [
                    x.to(device) if isinstance(x, torch.Tensor) else x
                    for x in context["get_inputs"]()
                ]
            except TimeoutError:
                raise
            except Exception as e:
                raise InfraError(f"Failed to prepare inputs: {e}") from e

            # User code: elapsed_ms records events, runs model, then syncs internally
            elapsed.append(backend.elapsed_ms(lambda: model(*inputs)))

        return elapsed
