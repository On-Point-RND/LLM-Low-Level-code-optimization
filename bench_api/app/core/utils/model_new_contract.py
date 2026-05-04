"""
KernelBench contract: ModelNew(*get_init_inputs()) must see hyperparameters like the reference Model.

Older agent code omitted storing init args on ModelNew. After exec() we subclass ModelNew and,
if the instance still has no _init_args / _init_kwargs after __init__, set them from the ctor
arguments (same order as get_init_inputs / reference Model.__init__).
"""


def patch_model_new_init_args(context: dict) -> None:
    if "ModelNew" not in context:
        return
    orig = context["ModelNew"]
    if not isinstance(orig, type):
        return
    if getattr(orig, "_bench_api_init_args_patched", False):
        return

    class _ModelNewProxy(orig):  # type: ignore[misc,valid-type]
        _bench_api_init_args_patched = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not hasattr(self, "_init_args"):
                object.__setattr__(self, "_init_args", args)
            if not hasattr(self, "_init_kwargs"):
                object.__setattr__(self, "_init_kwargs", kwargs)

    _ModelNewProxy.__name__ = getattr(orig, "__name__", "ModelNew")
    _ModelNewProxy.__qualname__ = getattr(orig, "__qualname__", "ModelNew")
    context["ModelNew"] = _ModelNewProxy
