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
                    feedback = f"Output mismatch"
                if feedback is not None:
                    correctness = False
                    correctness_information = feedback
                    break
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        correctness = False
        correctness_information = error_msg
        return correctness, correctness_information

    return correctness, correctness_information