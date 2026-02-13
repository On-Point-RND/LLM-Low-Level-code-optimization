import torch
from backends.backend_registry import register_backend, Backend
import os
from utils.correctness import execute_template
from utils.performance import time_execution_event_template
from config import arch_list

@register_backend('cuda')
class CudaBackend(Backend):
    def __init__(self):
        self.context = {}
        self.device = self.get_device()

    def get_device(self):
        return torch.device('cuda:0')

    def get_hardware_name(self):
        return torch.cuda.get_device_name(device=self.device)

    def compile(self, generated_code, op):
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)
        try:
            compile(generated_code, "<string>", "exec")
            exec(generated_code, self.context)
            return True, None
        except Exception as e:
            return False, str(e)

    def correctness_execution(self, ref_src):
        synchronize = torch.cuda.synchronize
        try:
            exec(ref_src, self.context)
        except Exception as e:
            raise RuntimeError(f"Failed to compile reference model: {str(e)}")
        return execute_template(synchronize, self.device, self.context)

    def time_execution(self, eval_target='ModelNew'):
        synchronize = torch.cuda.synchronize
        event_class = torch.cuda.Event
        return time_execution_event_template(self.context, self.device, synchronize, event_class, eval_target)

    def cleanup(self):
        del self.context
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device=self.device)
            torch.cuda.synchronize(
                device=self.device
            )  # Wait for all CUDA operations to complete
