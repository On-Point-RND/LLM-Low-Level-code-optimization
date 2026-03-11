from typing import Dict, Any, Optional
import os
import sys
import logging
from pathlib import Path

from app.core.bench_kernel_isolated import evaluate_kernel_isolated

logger = logging.getLogger(__name__)
from app.models import EvaluateResponse, PerformanceStats, BaselineStats, EvaluationTiming
from app.services.mlflow_service import log_evaluation_result, log_error_to_run
from app.services.baseline_service import get_single_baseline
from app.config import REFERENCE_DIR, MULTIKERNELBENCH_PATH
from app.integration import get_reference_path, get_dataset


def evaluate_function(
    function_code: str,
    function: str,
    language: str,
    torch_compile: bool = False,
    torch_compile_baseline: bool = False,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    num_trials: Optional[int] = None,
    num_warmup: Optional[int] = None,
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None,
    device_id: int = 0,
    mode: str = 'full',
    baseline_mean_ms: Optional[float] = None,
) -> EvaluateResponse:
    """
    Evaluate a kernel function.
    
    Args:
        function_code: The kernel code to evaluate
        function: Function name from dataset
        language: Language/backend name
        torch_compile: Whether to use torch.compile for the kernel
        torch_compile_baseline: Whether to use torch.compile for the baseline
        experiment_name: Optional MLflow experiment name (folder name)
        run_name: Optional MLflow run name (specific run name)
        num_trials: Optional number of performance measurement trials (default: 100)
        mode: Evaluation mode ('full', 'compilation', 'validation_benchmark')
        baseline_mean_ms: Optional baseline mean to use (avoids re-computation)
    
    Returns:
        EvaluateResponse with evaluation results
    """
    result = evaluate_kernel_isolated(
        function_code=function_code,
        function=function,
        language=language,
        torch_compile=torch_compile,
        torch_compile_baseline=torch_compile_baseline,
        num_trials=num_trials,
        num_warmup=num_warmup,
        batch_size=batch_size,
        dim=dim,
        input_dims=input_dims,
        device_id=device_id,
        mode=mode,
        baseline_mean_ms=baseline_mean_ms,
    )
    
    if result.get('system_error'):
        raise RuntimeError(result['error'])
    
    # Convert to response model
    performance = None
    if result.get('performance'):
        performance = PerformanceStats(**result['performance'])
    else:
        if result.get('performance_error'):
            logger.debug(f"Performance Error: {result['performance_error']}")

        if result.get('error'):
            logger.debug(f"General Error: {result['error']}")
    
    timing = None
    if result.get('timing'):
        timing = EvaluationTiming(**result['timing'])
    
    # Ensure compile_info or correctness_info are not None if they failed
    compile_info = result.get('compile_info')
    if result.get('compiled') is False and compile_info is None:
        compile_info = result.get('error') or "Compilation failed"
    
    correctness_info = result.get('correctness_info')
    if result.get('compiled') is True and result.get('correctness') is False and correctness_info is None:
        correctness_info = result.get('error') or "Correctness check failed"

    performance_error = result.get('performance_error')
    if result.get('compiled') is True and result.get('performance') is None and performance_error is None:
        performance_error = result.get('error')

    # Get baseline and calculate speedup
    baseline = None
    speedup = None
    baseline_function_code = None
    # Only get baseline if we're NOT in compilation-only mode
    # If we're in validation_benchmark or full mode, we'll need it.
    if mode != 'compilation':
        try:
            baseline_result = get_single_baseline(
                language, 
                function,
                batch_size=batch_size,
                dim=dim,
                input_dims=input_dims,
                torch_compile=torch_compile_baseline
            )

        if baseline_result and 'baseline' in baseline_result:
            baseline_data = baseline_result['baseline']
            if isinstance(baseline_data, dict) and 'mean' in baseline_data:
                baseline = BaselineStats(
                    mean=baseline_data['mean'],
                    std=baseline_data['std'],
                    min=baseline_data['min'],
                    max=baseline_data['max'],
                    num_trials=baseline_data.get('num_trials', 0)
                )
                # Calculate speedup: baseline_mean / current_mean
                # Calculate even if correctness is False, as long as performance is available
                if performance and performance.mean > 0:
                    speedup = baseline.mean / performance.mean
                
                # Read baseline/reference code
                try:
                    ref_src_path = get_reference_path(function)
                    if ref_src_path:
                        with open(ref_src_path, 'r', encoding='utf-8') as f:
                            baseline_function_code = f.read()
                except Exception as e:
                    logger.warning(f"Failed to read baseline code for {function}: {e}")

    except Exception as e:
        # If baseline can't be retrieved, just continue without it
        logger.warning(f"Failed to get baseline for {function} on {language}: {e}", exc_info=True)
    
    # Log to MLflow if enabled (if experiment_name or run_name provided)
    # Only log if it's a full run, or the final phase (validation_benchmark),
    # or if compilation failed in the compilation phase.
    should_log = (experiment_name or run_name) and (
        mode in ('full', 'validation_benchmark') or not result.get('compiled')
    )
    
    if should_log:
        baseline_dict = None
        if baseline:
            baseline_dict = {
                'mean': baseline.mean,
                'std': baseline.std,
                'min': baseline.min,
                'max': baseline.max,
                'num_trials': baseline.num_trials
            }
        mlflow_run_id = log_evaluation_result(
            experiment_name=experiment_name,
            run_name=run_name,
            function=function,
            language=language,
            hardware=result['hardware'],
            compiled=result['compiled'],
            correctness=result.get('correctness'),
            performance=result.get('performance'),
            torch_compile=torch_compile,
            torch_compile_baseline=torch_compile_baseline,
            compile_info=compile_info,
            correctness_info=correctness_info,
            speedup=speedup,
            baseline=baseline_dict
        )
        if mlflow_run_id:
            error_text = result.get('error')
            if error_text:
                log_error_to_run(mlflow_run_id, error_text)
    
    perf_str = f"{performance.mean:.3g}ms" if performance else "N/A"
    speedup_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
    timing_str = f"Timing: (comp: {timing.compilation:.2f}s, corr: {timing.correctness:.2f}s, perf: {timing.performance:.2f}s)" if timing else ""
    
    error_snippet = ""
    if not result['compiled']:
        err = compile_info or ""
        first_line = err.strip().split('\n')[0] if err else "Unknown compile error"
        error_snippet = f", CompileError: {first_line}"
    elif result.get('correctness') is False:
        err = correctness_info or ""
        first_line = err.strip().split('\n')[0] if err else "Unknown correctness error"
        error_snippet = f", CorrectnessError: {first_line}"

    logger.info(
        f"[EVAL] Function: {function}, Language: {language}, "
        f"Compiled: {result['compiled']}, Correctness: {result.get('correctness')}, "
        f"Performance: {perf_str}, Speedup: {speedup_str} {timing_str}{error_snippet}"
    )
    
    return EvaluateResponse(
        function=function,
        language=language,
        hardware=result['hardware'],
        compiled=result['compiled'],
        correctness=result.get('correctness'),
        performance=performance,
        timing=timing,
        compile_info=compile_info,
        correctness_info=correctness_info,
        run_name=run_name,
        baseline=baseline,
        speedup=speedup,
        function_code=function_code,
        baseline_function_code=baseline_function_code,
        performance_error=performance_error
    )

