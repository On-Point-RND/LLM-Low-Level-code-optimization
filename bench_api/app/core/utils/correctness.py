import torch
import app.config as config
from app.core.phases.common import InfraError


def _set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def run_correctness(backend, ref_src: str):
    """
    Exec ref_src into backend.context, then run correctness trials comparing
    Model (reference) vs ModelNew (generated). Returns (passed, info_str).
    Raises InfraError for infrastructure failures (our fault).
    """
    try:
        exec(ref_src, backend.context)
    except Exception as e:
        raise InfraError(f"Failed to load reference model: {e}") from e

    context = backend.context
    device = backend.get_device()

    try:
        init_inputs = context["get_init_inputs"]()
        init_inputs = [
            x.to(device=device) if isinstance(x, torch.Tensor) else x
            for x in init_inputs
        ]
    except Exception as e:
        raise InfraError(f"Failed to prepare init inputs: {e}") from e

    with torch.no_grad():
        try:
            _set_seed(config.SEED_NUM)
            original_model = context["Model"](*init_inputs).to(device)
            backend.synchronize()
        except Exception as e:
            raise InfraError(f"Failed to instantiate reference model: {e}") from e

        try:
            _set_seed(config.SEED_NUM)
            custom_model = context["ModelNew"](*init_inputs).to(device)
            backend.synchronize()
        except InfraError:
            raise
        except Exception as e:
            return False, backend.format_runtime_error(e)

    with torch.no_grad():
        for _ in range(config.NUM_CORRECT_TRIALS):
            try:
                inputs = context["get_inputs"]()
                inputs = [
                    x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs
                ]
                backend.synchronize()
            except Exception as e:
                raise InfraError(f"Failed to prepare inputs: {e}") from e

            try:
                ref_output = original_model(*inputs)
                backend.synchronize()
            except Exception as e:
                raise InfraError(f"Reference model execution failed: {e}") from e

            try:
                new_output = custom_model(*inputs)
                backend.synchronize()
            except InfraError:
                raise
            except Exception as e:
                return False, backend.format_runtime_error(e)

            if ref_output.shape != new_output.shape:
                return (
                    False,
                    f"Output shape mismatch: Expected {ref_output.shape}, got {new_output.shape}",
                )

            if not torch.allclose(ref_output, new_output, **backend.tolerances):
                diff = torch.abs(ref_output - new_output)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()
                is_close = torch.isclose(ref_output, new_output, **backend.tolerances)
                match_rate = 100.0 * torch.sum(is_close).item() / is_close.numel()
                return (
                    False,
                    f"Output mismatch: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, match_rate={match_rate:.2f}%",
                )

    return True, ""
