import tvm
from tvm import relax


def get_cuda_tir_pipeline(target="cuda"):
    """TIR pipeline with DefaultGPUSchedule for unscheduled primfuncs (e.g. from LLM or Relax FuseTIR)."""
    target_obj = tvm.target.Target(target)
    default_tir = tvm.tir.get_default_tir_pipeline(target_obj)
    return tvm.ir.transform.Sequential([
        tvm.tir.transform.DefaultGPUSchedule(),
        default_tir,
    ])


def build_relax_cuda(mod, target="cuda"):
    """Build Relax module for CUDA, applying DefaultGPUSchedule to unscheduled TIR."""
    target_obj = tvm.target.Target(target)
    return relax.build(mod, target=target_obj, tir_pipeline=get_cuda_tir_pipeline(target))
