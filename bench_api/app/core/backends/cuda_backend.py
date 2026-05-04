import torch
import os
import logging
from pathlib import Path
from app.core.backends.base_backend import Backend
from app.core.backends.backend_registry import register_backend
from app.core.utils.build_log import compact_build_log
from app.core.utils.model_new_contract import patch_model_new_init_args
from app.config import ARCH_LIST, CUDA_BUILD_ROOT

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
        import torch.utils.cpp_extension as _cpp_ext

        if "cpp_extension" not in generated_code:
            return False, (
                "CUDA submissions must use torch.utils.cpp_extension "
                "(load or load_inline) to implement ModelNew"
            )

        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        if self.arch_list:
            os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(self.arch_list)

        _called = []
        _orig_load, _orig_load_inline = _cpp_ext.load, _cpp_ext.load_inline

        def _patched_load(*args, **kwargs):
            _called.append(True)
            return _orig_load(*args, **kwargs)

        def _patched_load_inline(*args, **kwargs):
            _called.append(True)
            return _orig_load_inline(*args, **kwargs)

        _cpp_ext.load = _patched_load
        _cpp_ext.load_inline = _patched_load_inline
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
            patch_model_new_init_args(self.context)
        except OSError as e:
            if e.errno in (12, 28):
                raise
            raw = f"{type(e).__name__}: {str(e)}"
            return False, self.parse_compile_error(raw)
        except Exception as e:
            import torch
            if isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            raw = f"{type(e).__name__}: {str(e)}"
            return False, self.parse_compile_error(raw)
        finally:
            _cpp_ext.load = _orig_load
            _cpp_ext.load_inline = _orig_load_inline

        if not _called:
            return False, (
                "cpp_extension is imported but load/load_inline is never called — "
                "ModelNew must compile and load a CUDA kernel"
            )

        return True, None

    def parse_compile_error(self, raw: str) -> str:
        return compact_build_log(raw)

    @classmethod
    def get_subprocess_env(cls, device_id: int, benchmark_mode: bool, req_id: str = None) -> dict:
        env = {
            "CUDA_VISIBLE_DEVICES": str(device_id),
            "TORCH_USE_CUDA_DSA": None,
            "CUDA_LAUNCH_BLOCKING": None if benchmark_mode else "1",
        }
        if req_id:
            build_dir = CUDA_BUILD_ROOT / req_id
            build_dir.mkdir(parents=True, exist_ok=True)
            env["TORCH_EXTENSIONS_DIR"] = str(build_dir)
        return env

    @classmethod
    def cleanup_request(cls, req_id: str) -> None:
        import shutil
        shutil.rmtree(CUDA_BUILD_ROOT / req_id, ignore_errors=True)

    @classmethod
    def setup_server_env(cls, workspace_tmp: str) -> None:
        cuda_cache_dir = os.path.join(workspace_tmp, "cuda_cache")
        os.makedirs(cuda_cache_dir, exist_ok=True)
        if not os.getenv("CUDA_CACHE_PATH"):
            os.environ["CUDA_CACHE_PATH"] = cuda_cache_dir


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
        return {"atol": 1e-2, "rtol": 1e-2}

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
