BACKEND_REGISTRY = {}


def register_backend(name):
    def decorator(cls):
        try:
            BACKEND_REGISTRY[name] = cls()
        except Exception as e:
            print(f"[WARNING] Failed to instantiate backend {name}: {e}")
        return cls

    return decorator


def get_backend(name):
    if name not in BACKEND_REGISTRY:
        try:
            import importlib

            importlib.import_module(f"app.core.backends.{name}_backend")
        except ImportError:
            pass

    return BACKEND_REGISTRY.get(name)


def get_subprocess_env(language: str, device_id: int, benchmark_mode: bool) -> dict:
    backend = get_backend(language)
    if backend:
        return type(backend).get_subprocess_env(device_id, benchmark_mode)
    return {}
