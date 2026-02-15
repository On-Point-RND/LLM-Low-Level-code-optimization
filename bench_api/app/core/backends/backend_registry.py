BACKEND_REGISTRY = {}

def register_backend(name):
    def decorator(cls):
        # Instantiate the backend immediately upon registration, as per original design
        try:
            BACKEND_REGISTRY[name] = cls()
        except Exception as e:
            print(f"[WARNING] Failed to instantiate backend {name}: {e}")
        return cls
    return decorator

def get_backend(name):
    if name not in BACKEND_REGISTRY:
        # Try to import the module dynamically if not found
        try:
            import importlib
            importlib.import_module(f"app.core.backends.{name}_backend")
        except ImportError:
            pass
            
    return BACKEND_REGISTRY.get(name)
