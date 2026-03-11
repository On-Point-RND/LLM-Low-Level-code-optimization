import torch
import app.config as config


def _set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def run_correctness(backend, ref_src: str):
    """
    Exec ref_src into backend.context, then run correctness trials comparing
    Model (reference) vs ModelNew (generated). Returns (passed, info_str).
    """
    try:
        exec(ref_src, backend.context)
    except Exception as e:
        raise RuntimeError(f"Failed to compile reference model: {str(e)}")

    context = backend.context
    device = backend.get_device()

    get_inputs = context['get_inputs']
    get_init_inputs = context['get_init_inputs']
    Model = context['Model']
    ModelNew = context['ModelNew']

    try:
        init_inputs = get_init_inputs()
        init_inputs = [x.to(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs]

        with torch.no_grad():
            _set_seed(config.SEED_NUM)
            original_model = Model(*init_inputs).to(device)
            backend.synchronize()
            _set_seed(config.SEED_NUM)
            custom_model = ModelNew(*init_inputs).to(device)
            backend.synchronize()

        with torch.no_grad():
            for _ in range(config.NUM_CORRECT_TRIALS):
                inputs = get_inputs()
                inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
                backend.synchronize()
                ref_output = original_model(*inputs)
                backend.synchronize()
                new_output = custom_model(*inputs)
                backend.synchronize()

                if ref_output.shape != new_output.shape:
                    feedback = f"Output shape mismatch: Expected {ref_output.shape}, got {new_output.shape}"
                elif not torch.allclose(ref_output, new_output, **backend.tolerances):
                    diff = torch.abs(ref_output - new_output)
                    max_diff = torch.max(diff).item()
                    mean_diff = torch.mean(diff).item()
                    is_close = torch.isclose(ref_output, new_output, **backend.tolerances)
                    match_rate = 100.0 * torch.sum(is_close).item() / is_close.numel()
                    feedback = f"Output mismatch: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, match_rate={match_rate:.2f}%"
                else:
                    feedback = None

                if feedback is not None:
                    return False, feedback

    except Exception as e:
        import traceback
        import sys
        import os

        try:
            current_file = os.path.abspath(__file__)
            app_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
        except Exception:
            app_root = "/app"

        _, _, exc_traceback = sys.exc_info()
        full_tb = traceback.extract_tb(exc_traceback)

        filtered_tb = [
            frame for frame in full_tb
            if ("<" in os.path.abspath(frame.filename) and ">" in os.path.abspath(frame.filename))
            or not os.path.abspath(frame.filename).startswith(app_root)
        ]
        if not filtered_tb:
            filtered_tb = full_tb[-2:]

        formatted_tb = "".join(traceback.format_list(filtered_tb))
        error_msg = f"{type(e).__name__}: {str(e)}\n\nTraceback (most recent call last):\n{formatted_tb}"
        return False, error_msg

    return True, ''
