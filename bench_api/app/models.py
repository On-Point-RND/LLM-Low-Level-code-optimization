from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, Any


class BaselineRequest(BaseModel):
    language: str = Field(..., description="Language/backend (cuda, triton, ascendc, sycl, pallas, tilelang_ascend)")
    function: str = Field(..., description="Function name from dataset or 'all' to get all baselines for the language")
    torch_compile: bool = Field(False, description="Whether to use torch.compile for baseline")
    batch_size: Optional[int] = Field(None, description="Batch size for baseline evaluation (overrides reference file default)")
    dim: Optional[int] = Field(None, description="Dimension for baseline evaluation (overrides reference file default)")
    input_dims: Optional[Dict[str, Any]] = Field(None, description="Custom input dimensions as dict (e.g., {'batch_size': 32, 'dim': 8192, 'height': 224, 'width': 224})")
    experiment_name: Optional[str] = Field(None, description="MLflow experiment name for logging baseline")
    run_name: Optional[str] = Field(None, description="MLflow run name for logging baseline")


class BaselineStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    num_trials: int


class SingleBaselineResponse(BaseModel):
    function: str
    language: str
    hardware: str
    compute_capability: Optional[str] = Field(None, description="The compute capability of the hardware (e.g., '8.0' for CUDA)")
    baseline: BaselineStats
    cached: bool
    torch_compile: bool = Field(False, description="Whether torch.compile was used for baseline")
    function_code: Optional[str] = Field(None, description="The reference/baseline kernel code")
    batch_size: Optional[int] = Field(None, description="Batch size used for baseline evaluation")
    dim: Optional[int] = Field(None, description="Dimension used for baseline evaluation")
    input_dims: Optional[Dict[str, Any]] = Field(None, description="Custom input dimensions used for baseline evaluation")


class AllBaselinesResponse(BaseModel):
    function: str = "all"
    language: str
    hardware: str
    compute_capability: Optional[str] = Field(None, description="The compute capability of the hardware (e.g., '8.0' for CUDA)")
    baselines: Dict[str, BaselineStats]
    cached: bool
    torch_compile: bool = Field(False, description="Whether torch.compile was used for baselines")
    function_codes: Optional[Dict[str, str]] = Field(None, description="Reference/baseline kernel codes for each function")


class EvaluateRequest(BaseModel):
    torch_compile: bool = Field(False, description="Whether to use torch.compile for the submitted kernel")
    torch_compile_baseline: bool = Field(False, description="Whether to use torch.compile for baseline (reference) model")
    language: str = Field(..., description="Language/backend")
    function: str = Field(..., description="Function name")
    function_code: Optional[str] = Field(None, description="Kernel code content as string")
    function_code_file: Optional[str] = Field(None, description="Kernel code as base64 encoded string or file path")
    experiment_name: Optional[str] = Field(None, description="MLflow experiment name (folder name in MLflow)")
    run_name: Optional[str] = Field(None, description="MLflow run name (name of the specific run within the experiment)")
    num_trials: Optional[int] = Field(None, description="Number of performance measurement trials (default: 100)")
    num_warmup: Optional[int] = Field(None, description="Number of warmup iterations before timing (default: 3)")
    batch_size: Optional[int] = Field(None, description="Batch size for evaluation (overrides reference file default)")
    dim: Optional[int] = Field(None, description="Dimension for evaluation (overrides reference file default)")
    input_dims: Optional[Dict[str, Any]] = Field(None, description="Custom input dimensions as dict (e.g., {'batch_size': 32, 'dim': 8192, 'height': 224, 'width': 224})")
    
    @model_validator(mode='after')
    def validate_code_source(self):
        """Validate that either function_code or function_code_file is provided"""
        if not self.function_code and not self.function_code_file:
            raise ValueError("Either function_code or function_code_file must be provided")
        return self


class PerformanceStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    num_trials: int


class EvaluationTiming(BaseModel):
    compilation: Optional[float] = Field(None, description="Time spent on compilation in seconds")
    correctness: Optional[float] = Field(None, description="Time spent on correctness check in seconds")
    performance: Optional[float] = Field(None, description="Time spent on performance measurement in seconds")
    total: Optional[float] = Field(None, description="Total evaluation time in seconds (excluding baseline)")


class EvaluateResponse(BaseModel):
    function: str
    language: str
    hardware: str
    compute_capability: Optional[str] = Field(None, description="The compute capability of the hardware (e.g., '8.0' for CUDA)")
    compiled: bool
    correctness: Optional[bool] = None
    performance: Optional[PerformanceStats] = None
    timing: Optional[EvaluationTiming] = None
    compile_info: Optional[str] = None
    correctness_info: Optional[str] = None
    run_name: Optional[str] = Field(None, description="MLflow run name")
    baseline: Optional[BaselineStats] = Field(None, description="Baseline performance stats")
    speedup: Optional[float] = Field(None, description="Speedup relative to baseline (baseline_mean / current_mean)")
    function_code: Optional[str] = Field(None, description="The kernel code that was evaluated")
    baseline_function_code: Optional[str] = Field(None, description="The reference/baseline kernel code")
    performance_error: Optional[str] = Field(None, description="Error message if performance measurement failed")


class HelpResponse(BaseModel):
    languages: list[str]
    functions: Dict[str, str]  # function_name -> category
    categories: list[str]
    endpoints: Dict[str, Any]



