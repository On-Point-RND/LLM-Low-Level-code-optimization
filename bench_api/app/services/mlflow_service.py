"""
MLflow service for logging evaluation results.
"""
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional
import logging
import time

from app.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, ENABLE_MLFLOW, LOGS_DIR

logger = logging.getLogger(__name__)

_mlflow_initialized = False
_experiment_log_files = {}

def _ensure_mlflow_initialized():
    global _mlflow_initialized
    if not ENABLE_MLFLOW:
        return
    if _mlflow_initialized:
        return
    try:
        import socket
        from urllib.parse import urlparse
        parsed = urlparse(MLFLOW_TRACKING_URI)
        if parsed.hostname:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((parsed.hostname, parsed.port or 5050))
            sock.close()
            if result != 0:
                logger.warning(f"MLflow server at {MLFLOW_TRACKING_URI} is not reachable, MLflow logging may fail")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
        _mlflow_initialized = True
    except Exception as e:
        logger.warning(f"Failed to initialize MLflow: {e}")


def _get_or_create_experiment(experiment_name: str) -> Optional[str]:
    """
    Get or create MLflow experiment by name.
    
    Args:
        experiment_name: Name of the experiment (folder name in MLflow)
    
    Returns:
        Experiment ID or None if failed
    """
    if not ENABLE_MLFLOW:
        return None
    
    _ensure_mlflow_initialized()
    
    if not experiment_name or not experiment_name.strip():
        logger.warning("Empty experiment name provided")
        return None
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created MLflow experiment: {experiment_name} (id: {experiment_id})")
            try:
                mlflow.set_experiment(experiment_name)
            except Exception as e:
                logger.warning(f"Could not set experiment as active (non-critical): {e}")
        else:
            experiment_id = experiment.experiment_id
            if hasattr(experiment, 'lifecycle_stage') and experiment.lifecycle_stage == 'deleted':
                logger.warning(f"Experiment '{experiment_name}' is deleted. Attempting to restore...")
                try:
                    client = MlflowClient()
                    client.restore_experiment(experiment_id)
                    logger.info(f"Restored MLflow experiment: {experiment_name} (id: {experiment_id})")
                    try:
                        mlflow.set_experiment(experiment_name)
                    except Exception as e:
                        logger.warning(f"Could not set restored experiment as active (non-critical): {e}")
                except Exception as restore_error:
                    logger.error(f"Failed to restore deleted experiment '{experiment_name}': {restore_error}")
                    logger.info(f"Creating new experiment with name '{experiment_name}_new'")
                    new_name = f"{experiment_name}_new"
                    experiment_id = mlflow.create_experiment(new_name)
                    logger.info(f"Created new MLflow experiment: {new_name} (id: {experiment_id})")
            else:
                logger.debug(f"Using existing MLflow experiment: {experiment_name} (id: {experiment_id})")
                try:
                    mlflow.set_experiment(experiment_name)
                except Exception as e:
                    logger.warning(f"Could not set experiment as active (non-critical): {e}")
        return experiment_id
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"MLflow error setting up experiment '{experiment_name}': {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
    except Exception as e:
        logger.error(f"Failed to setup MLflow experiment '{experiment_name}': {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def log_evaluation_result(
    experiment_name: Optional[str],
    run_name: Optional[str],
    function: str,
    language: str,
    hardware: str,
    compiled: bool,
    correctness: Optional[bool],
    performance: Optional[Dict[str, Any]],
    torch_compile: bool,
    torch_compile_baseline: bool = False,
    compile_info: Optional[str] = None,
    correctness_info: Optional[str] = None,
    speedup: Optional[float] = None,
    baseline: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    if not ENABLE_MLFLOW:
        return None
    
    if not experiment_name:
        if MLFLOW_EXPERIMENT_NAME:
            target_experiment_name = MLFLOW_EXPERIMENT_NAME
        else:
            logger.warning("No experiment name provided and MLFLOW_EXPERIMENT_NAME not set")
            return None
    else:
        target_experiment_name = experiment_name.strip()
        if not target_experiment_name:
            logger.warning("Empty experiment name provided")
            return None
    
    experiment_id = _get_or_create_experiment(target_experiment_name)
    if experiment_id is None:
        logger.error(f"Failed to get or create experiment '{target_experiment_name}'")
        return None
    
    try:
        # Build run name: use provided run_name or generate from language/function/torch_compile
        if run_name:
            final_run_name = run_name
        else:
            tc_suffix = "tc_true" if torch_compile else "tc_false"
            final_run_name = f"{language}__{function}__{tc_suffix}__{int(time.time())}"
        
        with mlflow.start_run(experiment_id=experiment_id, run_name=final_run_name) as run:
            # Log parameters
            mlflow.log_param("function", function)
            mlflow.log_param("language", language)
            mlflow.log_param("hardware", hardware)
            mlflow.log_param("torch_compile", torch_compile)
            mlflow.log_param("torch_compile_baseline", torch_compile_baseline)
            mlflow.log_param("compiled", compiled)
            
            if run_name:
                mlflow.set_tag("run_name", run_name)
            if experiment_name:
                mlflow.set_tag("experiment_name", experiment_name)
            
            # Log metrics
            if compiled:
                mlflow.log_metric("compiled", 1.0)
            else:
                mlflow.log_metric("compiled", 0.0)
            
            if correctness is not None:
                mlflow.log_metric("correctness", 1.0 if correctness else 0.0)
            
            # Log performance metrics
            if performance:
                mlflow.log_metric("performance_mean", performance.get("mean", 0.0))
                mlflow.log_metric("performance_std", performance.get("std", 0.0))
                mlflow.log_metric("performance_min", performance.get("min", 0.0))
                mlflow.log_metric("performance_max", performance.get("max", 0.0))
                mlflow.log_metric("performance_num_trials", performance.get("num_trials", 0))
            
            if baseline:
                mlflow.log_metric("baseline_mean", baseline.get("mean", 0.0))
                mlflow.log_metric("baseline_std", baseline.get("std", 0.0))
                mlflow.log_metric("baseline_min", baseline.get("min", 0.0))
                mlflow.log_metric("baseline_max", baseline.get("max", 0.0))
                mlflow.log_metric("baseline_num_trials", baseline.get("num_trials", 0))
            
            if speedup is not None:
                mlflow.log_metric("speedup", speedup)
            
            # Log info as artifacts if available
            if compile_info:
                mlflow.log_text(compile_info, "compile_info.txt")
            
            if correctness_info:
                mlflow.log_text(correctness_info, "correctness_info.txt")
            
            logger.info(f"Logged evaluation to MLflow: experiment={target_experiment_name}, run_id={run.info.run_id}, run_name={final_run_name}")
            
            from datetime import datetime
            import json
            
            safe_experiment_name = target_experiment_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
            
            if target_experiment_name not in _experiment_log_files:
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                _experiment_log_files[target_experiment_name] = LOGS_DIR / f"{safe_experiment_name}_{timestamp_str}.jsonl"
            
            local_log_file = _experiment_log_files[target_experiment_name]
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "evaluation",
                "experiment_name": target_experiment_name,
                "run_name": final_run_name,
                "run_id": run.info.run_id,
                "function": function,
                "language": language,
                "hardware": hardware,
                "compiled": compiled,
                "correctness": correctness,
                "torch_compile": torch_compile,
                "torch_compile_baseline": torch_compile_baseline,
                "performance": performance,
                "baseline": baseline,
                "speedup": speedup,
                "compile_info": compile_info,
                "correctness_info": correctness_info
            }
            
            with open(local_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            logger.debug(f"Logged evaluation locally to {local_log_file}")
            
            return run.info.run_id
            
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {e}")
        return None


def log_error_to_run(run_id: str, error_text: str) -> bool:
    """
    Log error text to an existing MLflow run.

    Args:
        run_id: Existing MLflow run ID
        error_text: Full error message text

    Returns:
        True if successful, False otherwise
    """
    if not ENABLE_MLFLOW or not run_id or not error_text:
        return False

    try:
        client = MlflowClient()

        client.set_tag(run_id, "has_error", "true")
        client.set_tag(run_id, "error_message", error_text[:250])
        client.log_metric(run_id, "error_occurred", 1)

        try:
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(error_text)
                temp_file = f.name

            client.log_artifact(run_id, temp_file, "errors")

            try:
                os.unlink(temp_file)
            except Exception:
                pass

        except Exception as artifact_e:
            logger.warning(f"Failed to log error artifact for run {run_id}: {artifact_e}")

        logger.info(f"Logged error to existing MLflow run {run_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to log error to existing MLflow run {run_id}: {e}")
        return False
 

def log_baseline_result(
    experiment_name: Optional[str],
    run_name: Optional[str],
    function: str,
    language: str,
    hardware: str,
    baseline: Dict[str, Any],
    torch_compile: bool = False,
    function_code: Optional[str] = None,
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    if not ENABLE_MLFLOW:
        return None
    
    if not experiment_name:
        if MLFLOW_EXPERIMENT_NAME:
            target_experiment_name = MLFLOW_EXPERIMENT_NAME
        else:
            logger.warning("No experiment name provided for baseline logging")
            return None
    else:
        target_experiment_name = experiment_name.strip()
        if not target_experiment_name:
            logger.warning("Empty experiment name provided for baseline logging")
            return None
    
    experiment_id = _get_or_create_experiment(target_experiment_name)
    if experiment_id is None:
        logger.error(f"Failed to get or create experiment '{target_experiment_name}' for baseline")
        return None
    
    try:
        if not run_name:
            run_name = f"baseline_{function}_{language}_{int(time.time())}"
        
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            mlflow.set_tag("type", "baseline")
            mlflow.log_param("function", function)
            mlflow.log_param("language", language)
            mlflow.log_param("hardware", hardware)
            mlflow.log_param("torch_compile", torch_compile)
            
            if batch_size is not None:
                mlflow.log_param("batch_size", batch_size)
            if dim is not None:
                mlflow.log_param("dim", dim)
            if input_dims:
                mlflow.log_param("input_dims", str(input_dims))
            
            mlflow.log_metric("baseline_mean", baseline.get("mean", 0.0))
            mlflow.log_metric("baseline_std", baseline.get("std", 0.0))
            mlflow.log_metric("baseline_min", baseline.get("min", 0.0))
            mlflow.log_metric("baseline_max", baseline.get("max", 0.0))
            mlflow.log_metric("baseline_num_trials", baseline.get("num_trials", 0))
            
            if function_code:
                mlflow.log_text(function_code, "baseline_function_code.txt")
            
            logger.info(f"Logged baseline to MLflow: experiment={target_experiment_name}, run_id={run.info.run_id}, run_name={run_name}")
            
            from datetime import datetime
            import json
            
            safe_experiment_name = target_experiment_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
            
            if target_experiment_name not in _experiment_log_files:
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                _experiment_log_files[target_experiment_name] = LOGS_DIR / f"{safe_experiment_name}_{timestamp_str}.jsonl"
            
            local_log_file = _experiment_log_files[target_experiment_name]
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "baseline",
                "experiment_name": target_experiment_name,
                "run_name": run_name,
                "run_id": run.info.run_id,
                "function": function,
                "language": language,
                "hardware": hardware,
                "torch_compile": torch_compile,
                "baseline": baseline,
                "batch_size": batch_size,
                "dim": dim,
                "input_dims": input_dims
            }
            
            with open(local_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            logger.debug(f"Logged baseline locally to {local_log_file}")
            
            return run.info.run_id
            
    except Exception as e:
        logger.error(f"Failed to log baseline to MLflow: {e}")
        return None

