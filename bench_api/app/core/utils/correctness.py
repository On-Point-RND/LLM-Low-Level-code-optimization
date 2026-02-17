import torch
import app.config as config

def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def execute_template(synchronize, device, context):
    correctness = True
    correctness_information = ''

    get_inputs = context['get_inputs']
    get_init_inputs = context['get_init_inputs']
    Model = context['Model']
    ModelNew = context['ModelNew']
        
    try:
        init_inputs = get_init_inputs()
        init_inputs = [
            x.to(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]
        with torch.no_grad():
            set_seed(config.SEED_NUM)  # set seed for reproducible weights
            original_model = Model(*init_inputs).to(device)
            synchronize(device=device)
            set_seed(config.SEED_NUM)
            custom_model = ModelNew(*init_inputs).to(device)
            synchronize(device=device)
        with torch.no_grad():
            for trial in range(config.NUM_CORRECT_TRIALS):
                inputs = get_inputs()
                inputs = [
                    x.to(device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                synchronize(device=device)
                ref_output = original_model(*inputs)       
                synchronize(device=device)
                new_output = custom_model(*inputs)
                synchronize(device=device) # ensure all GPU operations are completed before checking results
                feedback = None
                if ref_output.shape != new_output.shape:
                    feedback = f"Output shape mismatch: Expected {ref_output.shape}, got {new_output.shape}"
                elif not torch.allclose(ref_output, new_output, atol=1e-04, rtol=1e-04):
                    diff = torch.abs(ref_output - new_output)
                    max_diff = torch.max(diff).item()
                    mean_diff = torch.mean(diff).item()
                    is_close = torch.isclose(ref_output, new_output, atol=1e-04, rtol=1e-04)
                    match_rate = 100.0 * torch.sum(is_close).item() / is_close.numel()
                    feedback = f"Output mismatch: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, match_rate={match_rate:.2f}%"
                if feedback is not None:
                    correctness = False
                    correctness_information = feedback
                    break
    except Exception as e:
        import traceback
        import sys
        import os

        # Determine the root 'app' directory dynamically to filter internal frames
        # current file is app/core/utils/correctness.py
        # up 3 levels gives .../app
        try:
            current_file = os.path.abspath(__file__)
            app_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
        except Exception:
            app_root = "/app" # Fallback if __file__ is weird
        
        _, _, exc_traceback = sys.exc_info()
        full_tb = traceback.extract_tb(exc_traceback)
        
        filtered_tb = []
        for frame in full_tb:
            # frame.filename is usually absolute
            filename = os.path.abspath(frame.filename)
            
            # Keep dynamic code (user kernels)
            if "<" in filename and ">" in filename: 
                 filtered_tb.append(frame)
            # Filter out internal service code
            elif not filename.startswith(app_root):
                 filtered_tb.append(frame)
        
        if not filtered_tb:
            filtered_tb = full_tb[-2:]
            
        formatted_tb = "".join(traceback.format_list(filtered_tb))
        error_msg = f"{type(e).__name__}: {str(e)}\n\nTraceback (most recent call last):\n{formatted_tb}"
        
        correctness = False
        correctness_information = error_msg
        return correctness, correctness_information

    return correctness, correctness_information