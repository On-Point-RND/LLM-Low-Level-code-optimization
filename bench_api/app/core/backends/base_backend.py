class Backend:
    def get_device(self):
        raise NotImplementedError

    def get_hardware_name(self):
        raise NotImplementedError

    def compile(self, generated_code, op):
        raise NotImplementedError

    def correctness_execution(self, ref_src):
        raise NotImplementedError

    def time_execution(self, eval_target='ModelNew'):
        # Support both modelNew and baseline eval
        raise NotImplementedError

    def is_available(self) -> bool:
        """Returns True if the backend is available on the current hardware."""
        return True

    def cleanup(self):
        pass
