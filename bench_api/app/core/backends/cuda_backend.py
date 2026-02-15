import torch
import os
from app.core.backends.base_backend import Backend
from app.core.backends.backend_registry import register_backend
from app.core.utils.correctness import execute_template
from app.core.utils.performance import time_execution_event_template
from app.config import ARCH_LIST

@register_backend('cuda')
class CudaBackend(Backend):
    def __init__(self):
        self.context = {}
        self.device = self.get_device()

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        return torch.device('cpu')

    def get_hardware_name(self):
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(device=self.device)
        return "CPU"

    def compile(self, generated_code, op):
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(ARCH_LIST)
        try:
            # We must compile/exec in the same context
            # We don't actually 'compile' Python code to binary here, we just exec it to load the function definitions
            compiled_code = compile(generated_code, "<string>", "exec")
            exec(compiled_code, self.context)
            return True, None
        except Exception as e:
            return False, str(e)

    def correctness_execution(self, ref_src):
        synchronize = torch.cuda.synchronize if torch.cuda.is_available() else lambda device=None: None
        try:
            exec(ref_src, self.context)
        except Exception as e:
            raise RuntimeError(f"Failed to compile reference model: {str(e)}")
        
        return execute_template(synchronize, self.device, self.context)

    def time_execution(self, eval_target='ModelNew'):
        synchronize = torch.cuda.synchronize if torch.cuda.is_available() else lambda device=None: None
        event_class = torch.cuda.Event if torch.cuda.is_available() else None
        
        if event_class is None:
             # Fallback for CPU timing if needed, though this backend is 'cuda'
             import time
             # Quick fallback implementation if running on CPU for some reason
             def cpu_time_execution(context, device, synchronize, event_class, eval_target):
                # Similar logic but using time.time()
                # Omitted for brevity unless needed, assuming CUDA is available for 'cuda' backend
                pass
             
        return time_execution_event_template(self.context, self.device, synchronize, event_class, eval_target)

    def cleanup(self):
        self.context = {} # Clear context
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=self.device)
                torch.cuda.synchronize(device=self.device)
