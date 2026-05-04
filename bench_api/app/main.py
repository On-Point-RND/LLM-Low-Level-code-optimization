import asyncio
import base64
import os
import logging
import uuid
from app.core.backends.backend_registry import cleanup_request as cleanup_backend_request

from fastapi import FastAPI, HTTPException

from app.models import (
    BaselineRequest,
    SingleBaselineResponse,
    AllBaselinesResponse,
    EvaluateRequest,
    EvaluateResponse,
    HelpResponse,
)
from app.services.baseline_service import get_single_baseline, get_all_baselines
from app.services.evaluation_service import (
    compile_kernel,
    validate_kernel,
    baseline_kernel,
    evaluate_kernel,
)
from app.services.help_service import get_help_info
from app.services.mlflow_service import log_baseline_result
from app.services.device_pool import DevicePool
from app.integration import register_all_bench_dirs
from app.core.backends.backend_registry import get_backend, setup_server_env
from app.config import DEVICE_IDS, BACKENDS

logger = logging.getLogger(__name__)

app = FastAPI(
    title="KernelBench Evaluation API",
    description="API for compiling, correctness-checking, and benchmarking CUDA kernels against KernelBench reference implementations.",
    version="1.0.0",
)

_device_pool: DevicePool = None


@app.on_event("startup")
async def startup_event():
    global _device_pool

    register_all_bench_dirs()

    workspace_tmp = os.getenv("WORKSPACE_TMP", "/workspace/tmp")
    os.makedirs(workspace_tmp, exist_ok=True)
    for key in ("TMPDIR", "TMP", "TEMP"):
        if not os.getenv(key):
            os.environ[key] = workspace_tmp

    for language in BACKENDS:
        setup_server_env(language, workspace_tmp)

    logger.info(
        f"TMPDIR={os.getenv('TMPDIR')}, CUDA_CACHE_PATH={os.getenv('CUDA_CACHE_PATH')}"
    )

    for language in BACKENDS:
        backend = get_backend(language)
        if backend and not backend.is_available():
            logger.warning(f"{language} backend is not available on this hardware")

    _device_pool = DevicePool(DEVICE_IDS)
    logger.info(f"Device pool initialized with devices {DEVICE_IDS}")


@app.on_event("shutdown")
async def shutdown_event():
    from app.core.bench_kernel_isolated import shutdown_all_workers
    shutdown_all_workers()
    logger.info("FastAPI application shutdown")


@app.get("/help", response_model=HelpResponse)
async def get_help():
    try:
        return HelpResponse(**get_help_info())
    except Exception as e:
        logger.exception("Failed to get help info")
        raise HTTPException(
            status_code=500, detail=f"Failed to get help info: {str(e)}"
        )


@app.post("/baseline", response_model=SingleBaselineResponse | AllBaselinesResponse)
async def get_baseline(request: BaselineRequest):
    logger.info(
        f"Received baseline request: function={request.function}, language={request.language}"
    )
    try:
        backend = get_backend(request.language)
        if backend and not backend.is_available():
            raise HTTPException(
                status_code=503,
                detail=f"Backend '{request.language}' is not available on this server hardware.",
            )

        if request.function == "all":
            result = get_all_baselines(
                request.language, torch_compile=request.torch_compile
            )
            return AllBaselinesResponse(
                function="all",
                language=request.language,
                hardware=result["hardware"],
                baselines=result["baselines"],
                cached=result["cached"],
                torch_compile=result["torch_compile"],
                function_codes=result.get("function_codes"),
            )
        else:
            result = get_single_baseline(
                request.language,
                request.function,
                batch_size=request.batch_size,
                dim=request.dim,
                input_dims=request.input_dims,
                torch_compile=request.torch_compile,
            )

            if request.experiment_name or request.run_name:
                log_baseline_result(
                    experiment_name=request.experiment_name,
                    run_name=request.run_name,
                    function=request.function,
                    language=request.language,
                    hardware=result["hardware"],
                    baseline=result["baseline"],
                    torch_compile=result["torch_compile"],
                    function_code=result.get("function_code"),
                    batch_size=result.get("batch_size"),
                    dim=result.get("dim"),
                    input_dims=result.get("input_dims"),
                )

            return SingleBaselineResponse(
                function=request.function,
                language=request.language,
                hardware=result["hardware"],
                baseline=result["baseline"],
                cached=result["cached"],
                torch_compile=result["torch_compile"],
                function_code=result.get("function_code"),
                batch_size=result.get("batch_size"),
                dim=result.get("dim"),
                input_dims=result.get("input_dims"),
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to get baseline")
        raise HTTPException(status_code=500, detail=f"Failed to get baseline: {str(e)}")


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    logger.info(
        f"Received evaluation request: function={request.function}, language={request.language}"
    )
    try:
        backend = get_backend(request.language)
        if backend and not backend.is_available():
            raise HTTPException(
                status_code=503,
                detail=f"Backend '{request.language}' is not available on this server hardware.",
            )

        if request.function_code:
            function_code = request.function_code
        elif request.function_code_file:
            try:
                function_code = base64.b64decode(request.function_code_file).decode("utf-8")
            except (ValueError, UnicodeDecodeError) as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"function_code_file must be valid base64-encoded UTF-8: {e}",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Either function_code or function_code_file must be provided",
            )

        req_id = uuid.uuid4().hex[:8]
        # Acquire a dedicated GPU device for the full pipeline.
        # All phases (compile, validate, baseline, benchmark) run on this same device.
        device_id = await _device_pool.acquire()
        logger.info(
            f"[GPU:{device_id}] ACQUIRE kernel={request.function} "
            f"req_id={req_id} free={_device_pool.depth}"
        )
        try:
            # Phase A: Compilation (on assigned device)
            compile_result = await asyncio.to_thread(
                compile_kernel,
                function_code=function_code,
                function=request.function,
                language=request.language,
                torch_compile=request.torch_compile or False,
                torch_compile_baseline=request.torch_compile_baseline or False,
                num_trials=request.num_trials,
                num_warmup=request.num_warmup,
                batch_size=request.batch_size,
                dim=request.dim,
                input_dims=request.input_dims,
                device_id=device_id,
                experiment_name=request.experiment_name,
                run_name=request.run_name,
                req_id=req_id,
            )

            if not compile_result.compiled:
                return compile_result

            # Phase B: Correctness validation (on assigned device)
            validate_result = await asyncio.to_thread(
                validate_kernel,
                function_code=function_code,
                function=request.function,
                language=request.language,
                torch_compile=request.torch_compile or False,
                torch_compile_baseline=request.torch_compile_baseline or False,
                num_trials=request.num_trials,
                num_warmup=request.num_warmup,
                batch_size=request.batch_size,
                dim=request.dim,
                input_dims=request.input_dims,
                device_id=device_id,
                req_id=req_id,
            )

            # Early exit: skip benchmark for incorrect kernels
            if not validate_result.correctness:
                return validate_result

            # Phase C: Baseline (only when explicitly requested)
            baseline = None
            if request.include_baseline:
                baseline = await asyncio.to_thread(
                    baseline_kernel,
                    function=request.function,
                    language=request.language,
                    torch_compile_baseline=request.torch_compile_baseline or False,
                    num_trials=request.num_trials,
                    num_warmup=request.num_warmup,
                    batch_size=request.batch_size,
                    dim=request.dim,
                    input_dims=request.input_dims,
                    device_id=device_id,
                    req_id=req_id,
                )
                if baseline is None:
                    return EvaluateResponse(
                        function=request.function,
                        language=request.language,
                        hardware=validate_result.hardware,
                        compiled=True,
                        correctness=validate_result.correctness,
                        correctness_info=validate_result.correctness_info,
                        timing=validate_result.timing,
                        performance_info=f"Baseline measurement failed for {request.function}",
                    )

            # Phase D: Benchmark + Profiling (compiled=True AND correct=True)
            return await asyncio.to_thread(
                evaluate_kernel,
                function_code=function_code,
                function=request.function,
                language=request.language,
                torch_compile=request.torch_compile or False,
                torch_compile_baseline=request.torch_compile_baseline or False,
                num_trials=request.num_trials,
                num_warmup=request.num_warmup,
                batch_size=request.batch_size,
                dim=request.dim,
                input_dims=request.input_dims,
                device_id=device_id,
                baseline_mean_ms=baseline.mean if baseline else None,
                baseline=baseline,
                correctness=validate_result.correctness,
                correctness_info=validate_result.correctness_info,
                experiment_name=request.experiment_name,
                run_name=request.run_name,
                req_id=req_id,
            )
        finally:
            _device_pool.release(device_id)
            logger.info(
                f"[GPU:{device_id}] RELEASE kernel={request.function} "
                f"req_id={req_id} free={_device_pool.depth}"
            )
            cleanup_backend_request(request.language, req_id)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to evaluate kernel")
        raise HTTPException(status_code=500, detail=f"Failed to evaluate: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "MultiKernelBench API",
        "version": "1.0.0",
        "endpoints": {
            "GET /help": "Get help information",
            "POST /baseline": "Get baseline performance",
            "POST /evaluate": "Evaluate kernel code",
        },
    }
