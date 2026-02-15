from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import base64
import os

from app.models import (
    BaselineRequest,
    SingleBaselineResponse,
    AllBaselinesResponse,
    EvaluateRequest,
    EvaluateResponse,
    HelpResponse
)
from app.services.baseline_service import get_single_baseline, get_all_baselines
from app.services.evaluation_service import evaluate_function
from app.services.help_service import get_help_info
from app.services.mlflow_service import log_baseline_result
from app.integration import register_kernelbench_dataset

app = FastAPI(
    title="MultiKernelBench API",
    description="API for evaluating kernel performance using MultiKernelBench",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    import logging
    logger = logging.getLogger(__name__)
    
    # Register KernelBench tasks
    register_kernelbench_dataset()
    
    workspace_tmp = os.getenv("WORKSPACE_TMP", "/workspace/tmp")
    os.makedirs(workspace_tmp, exist_ok=True)
    
    if not os.getenv("TMPDIR"):
        os.environ["TMPDIR"] = workspace_tmp
    if not os.getenv("TMP"):
        os.environ["TMP"] = workspace_tmp
    if not os.getenv("TEMP"):
        os.environ["TEMP"] = workspace_tmp
    
    cuda_cache_dir = os.path.join(workspace_tmp, "cuda_cache")
    os.makedirs(cuda_cache_dir, exist_ok=True)
    if not os.getenv("CUDA_CACHE_PATH"):
        os.environ["CUDA_CACHE_PATH"] = cuda_cache_dir
    
    logger.info(f"FastAPI application startup complete")
    logger.info(f"TMPDIR set to: {os.getenv('TMPDIR')}")
    logger.info(f"CUDA_CACHE_PATH set to: {os.getenv('CUDA_CACHE_PATH')}")
    
    # Check CUDA backend availability
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA detected. Initializing CudaBackend...")
            from app.core.backends.cuda_backend import CudaBackend
            # Initialize to trigger architecture detection
            _ = CudaBackend()
            logger.info("CudaBackend initialized successfully.")
        else:
            logger.warning("CUDA not detected. CUDA-based benchmarks will fail.")
    except Exception as e:
        logger.error(f"Failed to initialize CUDA backend: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    import logging
    logger = logging.getLogger(__name__)
    logger.info("FastAPI application shutdown")


@app.get("/help", response_model=HelpResponse)
async def get_help():
    """Get information about available languages, functions, and categories."""
    try:
        help_info = get_help_info()
        return HelpResponse(**help_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get help info: {str(e)}")


@app.post("/baseline", response_model=SingleBaselineResponse | AllBaselinesResponse)
async def get_baseline(request: BaselineRequest):
    """
    Get baseline performance for a function or all functions for a language.
    
    - If function is "all", returns all baselines for the language
    - Otherwise, returns baseline for the specific function
    """
    try:
        if request.function == "all":
            result = get_all_baselines(request.language)
            return AllBaselinesResponse(
                function="all",
                language=request.language,
                hardware=result['hardware'],
                baselines=result['baselines'],
                cached=result['cached'],
                function_codes=result.get('function_codes')
            )
        else:
            result = get_single_baseline(
                request.language, 
                request.function,
                batch_size=request.batch_size,
                dim=request.dim,
                input_dims=request.input_dims
            )
            
            if request.experiment_name or request.run_name:
                log_baseline_result(
                    experiment_name=request.experiment_name,
                    run_name=request.run_name,
                    function=request.function,
                    language=request.language,
                    hardware=result['hardware'],
                    baseline=result['baseline'],
                    function_code=result.get('function_code'),
                    batch_size=result.get('batch_size'),
                    dim=result.get('dim'),
                    input_dims=result.get('input_dims')
                )
            
            return SingleBaselineResponse(
                function=request.function,
                language=request.language,
                hardware=result['hardware'],
                baseline=result['baseline'],
                cached=result['cached'],
                function_code=result.get('function_code'),
                batch_size=result.get('batch_size'),
                dim=result.get('dim'),
                input_dims=result.get('input_dims')
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to get baseline: {str(e)}")


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    """
    Evaluate kernel code performance.
    
    Accepts JSON body with EvaluateRequest:
    - function_code: Kernel code as string (optional)
    - function_code_file: Kernel code as base64 encoded string or file path (optional)
    - One of function_code or function_code_file must be provided
    
    - Compiles the kernel code
    - Checks correctness against reference implementation
    - Measures performance
    - Optionally applies torch.compile if torch_compile=True
    - Returns run_name if provided for tracking
    """
    try:
        # Determine source of code
        if request.function_code:
            function_code = request.function_code
        elif request.function_code_file:
            # Try to decode as base64 first, if fails treat as file path
            try:
                function_code = base64.b64decode(request.function_code_file).decode('utf-8')
            except Exception:
                # If base64 decode fails, treat as file path
                if os.path.exists(request.function_code_file):
                    with open(request.function_code_file, 'r', encoding='utf-8') as f:
                        function_code = f.read()
                else:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"function_code_file is not valid base64 and file not found: {request.function_code_file}"
                    )
        else:
            raise HTTPException(status_code=400, detail="Either function_code or function_code_file must be provided")
        
        return evaluate_function(
            function_code=function_code,
            function=request.function,
            language=request.language,
            torch_compile=request.torch_compile if request.torch_compile is not None else False,
            experiment_name=request.experiment_name,
            run_name=request.run_name,
            num_trials=request.num_trials,
            batch_size=request.batch_size,
            dim=request.dim,
            input_dims=request.input_dims
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MultiKernelBench API",
        "version": "1.0.0",
        "endpoints": {
            "GET /help": "Get help information",
            "POST /baseline": "Get baseline performance",
            "POST /evaluate": "Evaluate kernel code"
        }
    }

