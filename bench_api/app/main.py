import asyncio
import base64
import os
import logging
import uuid

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
from app.integration import register_kernelbench_dataset
from app.core.backends.backend_registry import get_backend
from app.config import DEVICE_IDS

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MultiKernelBench API",
    description="API for evaluating kernel performance using MultiKernelBench",
    version="1.0.0",
)

_device_pool: DevicePool = None


@app.on_event("startup")
async def startup_event():
    global _device_pool

    register_kernelbench_dataset()

    workspace_tmp = os.getenv("WORKSPACE_TMP", "/workspace/tmp")
    os.makedirs(workspace_tmp, exist_ok=True)
    for key in ("TMPDIR", "TMP", "TEMP"):
        if not os.getenv(key):
            os.environ[key] = workspace_tmp

    cuda_cache_dir = os.path.join(workspace_tmp, "cuda_cache")
    os.makedirs(cuda_cache_dir, exist_ok=True)
    if not os.getenv("CUDA_CACHE_PATH"):
        os.environ["CUDA_CACHE_PATH"] = cuda_cache_dir

    logger.info(
        f"TMPDIR={os.getenv('TMPDIR')}, CUDA_CACHE_PATH={os.getenv('CUDA_CACHE_PATH')}"
    )

    try:
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA detected. Initializing CudaBackend...")
            from app.core.backends.cuda_backend import CudaBackend

            _ = CudaBackend()
            logger.info("CudaBackend initialized successfully.")
        else:
            logger.warning("CUDA not detected. CUDA-based benchmarks will fail.")
    except Exception as e:
        logger.error(f"Failed to initialize CUDA backend: {e}")

    _device_pool = DevicePool(DEVICE_IDS)
    logger.info(f"Device pool initialized with devices {DEVICE_IDS}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutdown")


@app.get("/help", response_model=HelpResponse)
async def get_help():
    try:
        return HelpResponse(**get_help_info())
    except Exception as e:
        import traceback

        traceback.print_exc()
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
        import traceback

        print(traceback.format_exc())
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
                function_code = base64.b64decode(request.function_code_file).decode(
                    "utf-8"
                )
            except Exception:
                if os.path.exists(request.function_code_file):
                    with open(request.function_code_file, "r", encoding="utf-8") as f:
                        function_code = f.read()
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"function_code_file is not valid base64 and file not found: {request.function_code_file}",
                    )
        else:
            raise HTTPException(
                status_code=400,
                detail="Either function_code or function_code_file must be provided",
            )

        req_id = uuid.uuid4().hex[:8]

        # Phase A: Compilation - No device lock
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
            device_id=0,
            experiment_name=request.experiment_name,
            run_name=request.run_name,
            req_id=req_id,
        )

        if not compile_result.compiled:
            return compile_result

        # Phase B & C: Validation and Benchmark - With device lock
        device_id = await _device_pool.acquire()
        logger.info(
            f"Acquired device {device_id} for {request.function}, ReqID: {req_id}"
        )
        try:
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
                raise RuntimeError(
                    f"Baseline measurement failed for {request.function}, cannot compute speedup"
                )

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
            logger.info(f"Released device {device_id}, ReqID: {req_id}")

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback

        traceback.print_exc()
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
