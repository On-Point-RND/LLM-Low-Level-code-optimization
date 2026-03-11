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

    def synchronize(self) -> None:
        """Block until all pending device operations complete."""
        pass

    def elapsed_ms(self, fn) -> float:
        """Run fn() once and return the device-accurate elapsed time in ms."""
        raise NotImplementedError

    @property
    def tolerances(self) -> dict:
        """Keyword args for torch.allclose during correctness checks."""
        return {'atol': 1e-4, 'rtol': 1e-4}

    def is_available(self) -> bool:
        return True

    def clear_device_memory(self) -> None:
        """Free device memory between evaluation stages (no context reset)."""
        pass

    def cleanup(self) -> None:
        """Full cleanup: reset context and free device memory."""
        pass
