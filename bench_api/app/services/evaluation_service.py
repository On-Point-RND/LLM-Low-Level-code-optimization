from typing import Dict, Any, Optional
import logging

from app.core.bench_kernel_isolated import run_baseline, run_compilation, run_validation, run_benchmark
from app.core.backends.backend_registry import get_backend
from app.models import EvaluateResponse, PerformanceStats, BaselineStats, EvaluationTiming
from app.services.mlflow_service import log_evaluation_result, log_error_to_run
from app.services.baseline_service import get_cached_baseline, save_cached_baseline
from app.integration import get_reference_path

logger = logging.getLogger(__name__)


def _build_worker_params(
    function_code: str,
    function: str,
    language: str,
    torch_compile: bool,
    torch_compile_baseline: bool,
    num_trials: Optional[int],
    num_warmup: Optional[int],
    batch_size: Optional[int],
    dim: Optional[int],
    input_dims: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        'function_code': function_code,
        'function': function,
        'language': language,
        'torch_compile': torch_compile,
        'torch_compile_baseline': torch_compile_baseline,
        'num_trials': num_trials,
        'num_warmup': num_warmup,
        'batch_size': batch_size,
        'dim': dim,
        'input_dims': input_dims,
    }


def compile_kernel(
    function_code: str,
    function: str,
    language: str,
    torch_compile: bool = False,
    torch_compile_baseline: bool = False,
    num_trials: Optional[int] = None,
    num_warmup: Optional[int] = None,
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None,
    device_id: int = 0,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    req_id: Optional[str] = None,
) -> EvaluateResponse:
    params = _build_worker_params(
        function_code, function, language, torch_compile, torch_compile_baseline,
        num_trials, num_warmup, batch_size, dim, input_dims,
    )
    result = run_compilation(params, device_id=device_id)

    if result.get('system_error'):
        raise RuntimeError(result['error'])

    timing = EvaluationTiming(**result['timing']) if result.get('timing') else None
    compile_info = result.get('compile_info')
    if not result.get('compiled') and compile_info is None:
        compile_info = result.get('error') or "Compilation failed"

    if not result.get('compiled') and (experiment_name or run_name):
        mlflow_run_id = log_evaluation_result(
            experiment_name=experiment_name,
            run_name=run_name,
            function=function,
            language=language,
            hardware=result['hardware'],
            compiled=False,
            correctness=None,
            performance=None,
            torch_compile=torch_compile,
            torch_compile_baseline=torch_compile_baseline,
            compile_info=compile_info,
            correctness_info=None,
            speedup=None,
            baseline=None,
        )
        if mlflow_run_id and result.get('error'):
            log_error_to_run(mlflow_run_id, result['error'])

    req_str = f", ReqID: {req_id}" if req_id else ""
    timing_str = f"Timing: (comp: {timing.compilation:.2f}s)" if timing and timing.compilation else ""
    error_snippet = ""
    if not result['compiled']:
        err = compile_info or ""
        first_line = err.strip().split('\n')[0] if err else "Unknown compile error"
        error_snippet = f", CompileError: {first_line}"

    logger.info(
        f"[COMPILE] Function: {function}, Language: {language}, "
        f"Compiled: {result['compiled']} {timing_str}{error_snippet}{req_str}"
    )

    return EvaluateResponse(
        function=function,
        language=language,
        hardware=result['hardware'],
        compiled=result['compiled'],
        timing=timing,
        compile_info=compile_info,
        function_code=function_code,
    )


def baseline_kernel(
    function: str,
    language: str,
    torch_compile_baseline: bool = False,
    num_trials: Optional[int] = None,
    num_warmup: Optional[int] = None,
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None,
    device_id: int = 0,
    req_id: Optional[str] = None,
) -> Optional[BaselineStats]:
    backend = get_backend(language)
    hardware = backend.get_hardware_name() if backend else 'unknown'

    cached = get_cached_baseline(
        language, function, hardware, batch_size=batch_size, dim=dim,
        input_dims=input_dims, torch_compile=torch_compile_baseline,
        num_trials=num_trials, num_warmup=num_warmup,
    )

    req_str = f", ReqID: {req_id}" if req_id else ""

    if cached:
        logger.info(
            f"[BASELINE] Function: {function}, Language: {language}, Device: {device_id}, "
            f"Performance: {cached['mean']:.3g}ms, Cached: True{req_str}"
        )
        return BaselineStats(
            mean=cached['mean'], std=cached['std'],
            min=cached['min'], max=cached['max'],
            num_trials=cached.get('num_trials', 0),
        )

    params = {
        'function_code': '',
        'function': function,
        'language': language,
        'torch_compile': False,
        'torch_compile_baseline': torch_compile_baseline,
        'num_trials': num_trials,
        'num_warmup': num_warmup,
        'batch_size': batch_size,
        'dim': dim,
        'input_dims': input_dims,
    }
    result = run_baseline(params, device_id=device_id)

    if result.get('system_error') or not result.get('baseline'):
        logger.warning(f"[BASELINE] Failed for {function}: {result.get('error')}{req_str}")
        return None

    entry = result['baseline']
    save_cached_baseline(
        language, result['hardware'], function, entry,
        batch_size=batch_size, dim=dim, input_dims=input_dims,
        torch_compile=torch_compile_baseline, num_trials=num_trials, num_warmup=num_warmup,
    )

    logger.info(
        f"[BASELINE] Function: {function}, Language: {language}, Device: {device_id}, "
        f"Performance: {entry['mean']:.3g}ms, Cached: False{req_str}"
    )
    return BaselineStats(
        mean=entry['mean'], std=entry['std'],
        min=entry['min'], max=entry['max'],
        num_trials=entry.get('num_trials', 0),
    )


def validate_kernel(
    function_code: str,
    function: str,
    language: str,
    torch_compile: bool = False,
    torch_compile_baseline: bool = False,
    num_trials: Optional[int] = None,
    num_warmup: Optional[int] = None,
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None,
    device_id: int = 0,
    req_id: Optional[str] = None,
) -> EvaluateResponse:
    params = _build_worker_params(
        function_code, function, language, torch_compile, torch_compile_baseline,
        num_trials, num_warmup, batch_size, dim, input_dims,
    )
    result = run_validation(params, device_id=device_id)

    if result.get('system_error'):
        raise RuntimeError(result['error'])

    if not result['compiled']:
        raise RuntimeError(f"Recompilation failed during validation for {function}: {result.get('error')}")

    timing = EvaluationTiming(**result['timing']) if result.get('timing') else None
    correctness_info = result.get('correctness_info')
    if result.get('correctness') is False and correctness_info is None:
        correctness_info = result.get('error') or "Correctness check failed"

    req_str = f", ReqID: {req_id}" if req_id else ""
    corr_timing = f" Timing: (corr: {timing.correctness:.2f}s)" if timing and timing.correctness else ""
    error_snippet = ""
    if result.get('correctness') is False:
        err = correctness_info or ""
        first_line = err.strip().split('\n')[0] if err else "Unknown correctness error"
        error_snippet = f", CorrectnessError: {first_line}"

    logger.info(
        f"[VALIDATE] Function: {function}, Language: {language}, Device: {device_id}, "
        f"Correctness: {result.get('correctness')}{corr_timing}{error_snippet}{req_str}"
    )

    return EvaluateResponse(
        function=function,
        language=language,
        hardware=result['hardware'],
        compiled=result['compiled'],
        correctness=result.get('correctness'),
        timing=timing,
        correctness_info=correctness_info,
        function_code=function_code,
    )


def evaluate_kernel(
    function_code: str,
    function: str,
    language: str,
    torch_compile: bool = False,
    torch_compile_baseline: bool = False,
    num_trials: Optional[int] = None,
    num_warmup: Optional[int] = None,
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None,
    device_id: int = 0,
    baseline_mean_ms: Optional[float] = None,
    baseline: Optional[BaselineStats] = None,
    correctness: Optional[bool] = None,
    correctness_info: Optional[str] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    req_id: Optional[str] = None,
) -> EvaluateResponse:
    params = _build_worker_params(
        function_code, function, language, torch_compile, torch_compile_baseline,
        num_trials, num_warmup, batch_size, dim, input_dims,
    )
    result = run_benchmark(params, device_id=device_id, baseline_mean_ms=baseline_mean_ms)

    if result.get('system_error'):
        raise RuntimeError(result['error'])

    performance = PerformanceStats(**result['performance']) if result.get('performance') else None
    timing = EvaluationTiming(**result['timing']) if result.get('timing') else None

    performance_error = result.get('performance_error')
    if performance is None and performance_error is None:
        performance_error = result.get('error')

    speedup = None
    if performance and performance.mean > 0:
        speedup = baseline.mean / performance.mean

    baseline_function_code = None
    try:
        ref_src_path = get_reference_path(function)
        if ref_src_path:
            with open(ref_src_path, 'r', encoding='utf-8') as f:
                baseline_function_code = f.read()
    except Exception as e:
        logger.warning(f"Failed to read baseline code for {function}: {e}")

    if experiment_name or run_name:
        baseline_dict = None
        if baseline:
            baseline_dict = {
                'mean': baseline.mean, 'std': baseline.std,
                'min': baseline.min, 'max': baseline.max,
                'num_trials': baseline.num_trials,
            }
        mlflow_run_id = log_evaluation_result(
            experiment_name=experiment_name,
            run_name=run_name,
            function=function,
            language=language,
            hardware=result['hardware'],
            compiled=True,
            correctness=correctness,
            performance=result.get('performance'),
            torch_compile=torch_compile,
            torch_compile_baseline=torch_compile_baseline,
            compile_info=None,
            correctness_info=correctness_info,
            speedup=speedup,
            baseline=baseline_dict,
        )
        if mlflow_run_id and result.get('error'):
            log_error_to_run(mlflow_run_id, result['error'])

    perf_str = f"{performance.mean:.3g}ms" if performance else "N/A"
    speedup_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
    req_str = f", ReqID: {req_id}" if req_id else ""
    perf_timing = f" Timing: (perf: {timing.performance:.2f}s)" if timing and timing.performance else ""
    perf_error_str = ""
    if performance_error:
        first_line = performance_error.strip().split('\n')[0]
        perf_error_str = f", PerfError: {first_line}"

    logger.info(
        f"[BENCH] Function: {function}, Language: {language}, Device: {device_id}, "
        f"Performance: {perf_str}, Speedup: {speedup_str}{perf_timing}{perf_error_str}{req_str}"
    )

    return EvaluateResponse(
        function=function,
        language=language,
        hardware=result['hardware'],
        compiled=True,
        correctness=correctness,
        performance=performance,
        timing=timing,
        correctness_info=correctness_info,
        run_name=run_name,
        baseline=baseline,
        speedup=speedup,
        function_code=function_code,
        baseline_function_code=baseline_function_code,
        performance_error=performance_error,
    )
