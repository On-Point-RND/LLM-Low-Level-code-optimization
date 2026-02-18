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
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    num_trials: Optional[int] = None,
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None,
) -> EvaluateResponse:
    """
    Evaluate a kernel function.
    
    Args:
        function_code: The kernel code to evaluate
        function: Function name from dataset
        language: Language/backend name
        torch_compile: Whether to use torch.compile
        experiment_name: Optional MLflow experiment name (folder name)
        run_name: Optional MLflow run name (specific run name)
        num_trials: Optional number of performance measurement trials (default: 100)
    
    Returns:
        EvaluateResponse with evaluation results
    """
    result = evaluate_kernel_isolated(
        function_code=function_code,
        function=function,
        language=language,
        torch_compile=torch_compile,
        num_trials=num_trials,
        batch_size=batch_size,
        dim=dim,
        input_dims=input_dims
    )
    
    # If baseline computation failed inside the kernel evaluation, raise an error to return HTTP 500
    if result.get('error'):
        if "Baseline computation failed" in str(result['error']):
             raise RuntimeError(result['error'])
        if result.get('stage') in ['initialization', 'baseline']:
             raise RuntimeError(f"System error during {result['stage']}: {result['error']}")
    
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
    
    # Get baseline and calculate speedup
    baseline = None
    speedup = None
    baseline_function_code = None
    try:
        baseline_result = get_single_baseline(
            language, 
            function,
            batch_size=batch_size,
            dim=dim,
            input_dims=input_dims
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
    if experiment_name or run_name:
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
            compile_info=result.get('compile_info'),
            correctness_info=result.get('correctness_info'),
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
        err = result.get('compile_info') or result.get('error') or ""
        first_line = err.strip().split('\n')[0] if err else "Unknown compile error"
        error_snippet = f", CompileError: {first_line}"
    elif result.get('correctness') is False:
        err = result.get('correctness_info') or result.get('error') or ""
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
        compile_info=result.get('compile_info'),
        correctness_info=result.get('correctness_info'),
        run_name=run_name,
        baseline=baseline,
        speedup=speedup,
        function_code=function_code,
        baseline_function_code=baseline_function_code
    )

