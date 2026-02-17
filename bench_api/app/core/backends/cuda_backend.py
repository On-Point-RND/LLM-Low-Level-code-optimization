import torch
import os
import logging
from app.core.backends.base_backend import Backend
from app.core.backends.backend_registry import register_backend
from app.core.utils.correctness import execute_template
from app.core.utils.performance import time_execution_event_template
from app.config import ARCH_LIST

logger = logging.getLogger(__name__)

@register_backend('cuda')
class CudaBackend(Backend):
    def __init__(self):
        self.context = {}
        self.device = self.get_device()
        
        # Load config
        self.arch_list = ARCH_LIST
        if torch.cuda.is_available():
            if not self.arch_list:
                try:
                    # Auto-detect architecture: (8, 9) -> "8.9"
                    capability = self.get_compute_capability()
                    if capability:
                        arch = f"{capability[0]}.{capability[1]}"
                        self.arch_list = [arch]
                        logger.info(f"Auto-detected CUDA architecture: {arch}")
                except Exception as e:
                    logger.warning(f"Failed to detect CUDA architecture: {e}")
                    self.arch_list = ["8.0"] # Fallback to Ampere
            else:
                 logger.info(f"Using configured CUDA architecture: {self.arch_list}")
        else:
             logger.warning("CUDA is not available. CudaBackend functionality will be limited.")

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        return torch.device('cpu')

    def get_hardware_name(self):
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(device=self.device)
        return "CPU"

    def get_compute_capability(self):
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(self.device)
        return None

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def compile(self, generated_code, op):
        import hashlib
        import linecache
        
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        if self.arch_list:
            os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(self.arch_list)
        
        try:
            # Generate a unique fake filename for the code
            # This allows traceback to show source lines without writing to disk
            fake_fname = f"<cuda_code_{hashlib.md5(generated_code.encode()).hexdigest()[:8]}>"
            
            # Register the source code in linecache
            linecache.cache[fake_fname] = (
                len(generated_code),
                None,
                generated_code.splitlines(True),
                fake_fname,
            )
            
            # Compile with the fake filename
            compiled_code = compile(generated_code, fake_fname, "exec")
            exec(compiled_code, self.context)
            return True, None
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)}"

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
        
        return time_execution_event_template(self.context, self.device, synchronize, event_class, eval_target)

    def cleanup(self):
        self.context = {} # Clear context
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=self.device)
                torch.cuda.synchronize(device=self.device)
