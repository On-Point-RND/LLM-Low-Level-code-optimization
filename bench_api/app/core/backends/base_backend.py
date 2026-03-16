from typing import Optional, Tuple


class Backend:
    context: dict  # execution namespace, populated by compile() and load_reference()

    def get_device(self):
        raise NotImplementedError

    def get_hardware_name(self) -> str:
        raise NotImplementedError

    def get_compute_capability(self) -> Optional[Tuple[int, int]]:
        """Returns (major, minor) compute capability, or None if not applicable."""
        return None

    def compile(self, generated_code: str, op: str) -> Tuple[bool, Optional[str]]:
        """Compile/exec kernel code into self.context. Returns (success, error)."""
        raise NotImplementedError

    def parse_compile_error(self, raw: str) -> str:
        """Extract the relevant lines from a raw compilation error string."""
        return raw

    def format_runtime_error(self, e: Exception) -> str:
        """Format a runtime exception, filtering traceback to user code frames."""
        import traceback
        import os

        try:
            app_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        except Exception:
            app_root = ""

        full_tb = traceback.extract_tb(e.__traceback__)
        filtered_tb = [
            frame
            for frame in full_tb
            if ("<" in frame.filename and ">" in frame.filename)
            or (app_root and not os.path.abspath(frame.filename).startswith(app_root))
        ]
        if not filtered_tb:
            filtered_tb = full_tb[-2:]

        return f"{type(e).__name__}: {e}\n\nTraceback (most recent call last):\n{''.join(traceback.format_list(filtered_tb))}"

    def synchronize(self) -> None:
        """Block until all pending device operations complete."""
        pass

    def elapsed_ms(self, fn) -> float:
        """Run fn() once and return the device-accurate elapsed time in ms."""
        raise NotImplementedError

    @property
    def tolerances(self) -> dict:
        """Keyword args for torch.allclose during correctness checks."""
        return {"atol": 1e-4, "rtol": 1e-4}

    @classmethod
    def get_subprocess_env(cls, device_id: int, benchmark_mode: bool, req_id: str = None) -> dict:
        """Env vars to apply to the subprocess before fork. None value means unset."""
        return {}

    @classmethod
    def cleanup_request(cls, req_id: str) -> None:
        """Clean up any per-request resources created by get_subprocess_env."""
        pass

    @classmethod
    def setup_server_env(cls, workspace_tmp: str) -> None:
        """Set process-level env vars needed by this backend at server startup."""
        pass

    def is_available(self) -> bool:
        return True

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible input generation."""
        pass

    def is_fatal_error(self, error_info: str) -> bool:
        """Return True if the error requires full backend cleanup (e.g. CUDA context corruption)."""
        return False

    def clear_device_memory(self) -> None:
        """Free device memory between evaluation stages (no context reset)."""
        pass

    def cleanup(self) -> None:
        """Full cleanup: reset context and free device memory."""
        pass
