import torch
import os
import logging
from app.core.backends.base_backend import Backend
from app.core.backends.backend_registry import register_backend
from app.core.utils.build_log import compact_build_log
from app.config import ARCH_LIST

logger = logging.getLogger(__name__)


@register_backend("cuda")
class CudaBackend(Backend):
    def __init__(self):
        self.context = {}
        self._device = self.get_device()

        self.arch_list = ARCH_LIST
        if torch.cuda.is_available():
            if not self.arch_list:
                try:
                    capability = self.get_compute_capability()
                    if capability:
                        arch = f"{capability[0]}.{capability[1]}"
                        self.arch_list = [arch]
                        logger.info(f"Auto-detected CUDA architecture: {arch}")
                except Exception as e:
                    logger.warning(f"Failed to detect CUDA architecture: {e}")
                    self.arch_list = ["8.0"]
            else:
                logger.info(f"Using configured CUDA architecture: {self.arch_list}")
        else:
            logger.warning(
                "CUDA is not available. CudaBackend functionality will be limited."
            )

    def get_device(self):
        return (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

    def get_hardware_name(self) -> str:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(device=self._device)
        return "CPU"

    def get_compute_capability(self):
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(self._device)
        return None

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def compile(self, generated_code: str, op: str):
        import hashlib
        import linecache

        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        if self.arch_list:
            os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(self.arch_list)

        try:
            fake_fname = (
                f"<cuda_code_{hashlib.md5(generated_code.encode()).hexdigest()[:8]}>"
            )
            linecache.cache[fake_fname] = (
                len(generated_code),
                None,
                generated_code.splitlines(True),
                fake_fname,
            )
            compiled_code = compile(generated_code, fake_fname, "exec")
            exec(compiled_code, self.context)
            return True, None
        except Exception as e:
            raw = f"{type(e).__name__}: {str(e)}"
            return False, self.parse_compile_error(raw)

    def parse_compile_error(self, raw: str) -> str:
        return compact_build_log(raw)

    @classmethod
    def get_subprocess_env(cls, device_id: int, benchmark_mode: bool) -> dict:
        return {
            "CUDA_VISIBLE_DEVICES": str(device_id),
            "TORCH_USE_CUDA_DSA": None,
            "CUDA_LAUNCH_BLOCKING": None if benchmark_mode else "1",
        }

    def set_seed(self, seed: int) -> None:
        torch.cuda.manual_seed(seed)

    def is_fatal_error(self, error_info: str) -> bool:
        lower = error_info.lower()
        return "cuda error" in lower or "illegal memory access" in lower

    def synchronize(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize(device=self._device)

    def elapsed_ms(self, fn) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        self.synchronize()
        return start.elapsed_time(end)

    @property
    def tolerances(self) -> dict:
        return {"atol": 1e-4, "rtol": 1e-4}

    def clear_device_memory(self) -> None:
        if torch.cuda.is_available():
            with torch.cuda.device(self._device):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def cleanup(self) -> None:
        self.context = {}
        if torch.cuda.is_available():
            with torch.cuda.device(self._device):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=self._device)
                torch.cuda.synchronize()
