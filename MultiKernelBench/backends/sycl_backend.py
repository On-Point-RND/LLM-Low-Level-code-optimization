import torch
from backends.backend_registry import register_backend, Backend
import os
from utils.correctness import execute_template
from utils.performance import time_execution_event_template
from config import arch_list_xpu

@register_backend('sycl')
class SyclBackend(Backend):
    def __init__(self):
        self.context = {}
        self.device = self.get_device()

    def get_device(self):
        return torch.device('xpu:0')

    def get_hardware_name(self):
        return torch.xpu.get_device_name(device=self.device)

    def compile(self, generated_code, op):
        os.environ["TORCH_XPU_ARCH_LIST"] = ";".join(arch_list_xpu)
        try:
            compile(generated_code, "<string>", "exec")
            exec(generated_code, self.context)
            return True, None
        except Exception as e:
            return False, str(e)

    def correctness_execution(self, ref_src):
        synchronize = torch.xpu.synchronize
        try:
            exec(ref_src, self.context)
        except Exception as e:
            raise RuntimeError(f"Failed to compile reference model: {str(e)}")
        return execute_template(synchronize, self.device, self.context)

    def time_execution(self, eval_target='ModelNew'):
        synchronize = torch.xpu.synchronize
        event_class = torch.xpu.Event
        return time_execution_event_template(self.context, self.device, synchronize, event_class, eval_target)

    def cleanup(self):
        del self.context
        with torch.xpu.device(self.device):
            torch.xpu.empty_cache()
            torch.xpu.reset_peak_memory_stats(device=self.device)
            torch.xpu.synchronize(
                device=self.device
            )
