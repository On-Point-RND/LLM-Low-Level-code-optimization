from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CompilationResult:
    hardware: str
    compiled: bool
    compute_capability: Optional[str] = None
    compile_info: Optional[str] = None
    timing: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    hardware: str
    compiled: bool
    correctness: Optional[bool]
    compute_capability: Optional[str] = None
    correctness_info: Optional[str] = None
    compile_info: Optional[str] = None
    timing: Optional[Dict[str, Any]] = None


@dataclass
class BaselineResult:
    hardware: str
    compute_capability: Optional[str] = None
    baseline: Optional[Dict[str, Any]] = None
    timing: Optional[Dict[str, Any]] = None
    system_error: Optional[str] = None


@dataclass
class BenchmarkResult:
    hardware: str
    compute_capability: Optional[str] = None
    performance: Optional[Dict[str, Any]] = None
    performance_info: Optional[str] = None
    timing: Optional[Dict[str, Any]] = None
    profiling_mode: Optional[str] = None   # "full" or "fast"
    cuda_profile: Optional[list] = None    # top CUDA kernels from torch.profiler
