import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from app.config import BASELINES_DIR, MULTIKERNELBENCH_PATH
from app.core.phases.baseline import compute_baseline, compute_all_baselines
from app.core.backends.backend_registry import get_backend
from app.integration import get_reference_path

logger = logging.getLogger(__name__)

# Import dataset to get function category
sys.path.insert(0, str(MULTIKERNELBENCH_PATH))
from dataset import dataset


def get_baseline_file_path(
    language: str,
    hardware: str,
    dims_key: Optional[str] = None,
    torch_compile: bool = False,
) -> Path:
    """Get the path to the baseline file for a language and hardware."""
    suffix = ""
    if torch_compile:
        suffix = "_compiled"

    if dims_key:
        return BASELINES_DIR / f"{language}_{hardware}_{dims_key}{suffix}.json"
    return BASELINES_DIR / f"{language}_{hardware}{suffix}.json"


def _create_dims_key(
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Create a key for dimensions to use in cache filename."""
    if not batch_size and not dim and not input_dims:
        return None

    parts = []
    if batch_size is not None:
        parts.append(f"bs{batch_size}")
    if dim is not None:
        parts.append(f"dim{dim}")
    if input_dims:
        # Sort keys for consistent ordering
        sorted_dims = sorted(input_dims.items())
        dims_str = "_".join(f"{k}{v}" for k, v in sorted_dims)
        parts.append(dims_str)

    return "_".join(parts) if parts else None


def load_baseline_file(
    language: str,
    hardware: str,
    dims_key: Optional[str] = None,
    torch_compile: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load baseline file if it exists."""
    baseline_path = get_baseline_file_path(language, hardware, dims_key, torch_compile)
    if baseline_path.exists():
        try:
            with open(baseline_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load baseline file {baseline_path}: {e}")
    return None


def save_baseline_file(
    language: str,
    hardware: str,
    data: Dict[str, Any],
    dims_key: Optional[str] = None,
    torch_compile: bool = False,
):
    """Save baseline data to file."""
    baseline_path = get_baseline_file_path(language, hardware, dims_key, torch_compile)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(data, f, indent=2)


def _read_reference_code(function: str) -> Optional[str]:
    """Read reference code for a function."""
    try:
        ref_src_path = get_reference_path(function)
        if ref_src_path:
            with open(ref_src_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        logger.warning(f"Failed to read reference code for {function}: {e}")
    return None


def _build_cache_key(
    function: str,
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None,
    num_trials: Optional[int] = None,
    num_warmup: Optional[int] = None,
    device_id: Optional[int] = None,
) -> tuple:
    """Returns (dims_key, cache_key)."""
    dims_key = _create_dims_key(batch_size, dim, input_dims)
    cache_key = function
    if dims_key:
        cache_key = f"{function}_{dims_key}"
    if num_trials is not None:
        cache_key = f"{cache_key}_t{num_trials}"
    if num_warmup is not None:
        cache_key = f"{cache_key}_w{num_warmup}"
    if device_id is not None:
        cache_key = f"{cache_key}_d{device_id}"
    return dims_key, cache_key


def get_cached_baseline(
    language: str,
    function: str,
    hardware: str,
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None,
    torch_compile: bool = False,
    num_trials: Optional[int] = None,
    num_warmup: Optional[int] = None,
    device_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    dims_key, cache_key = _build_cache_key(
        function, batch_size, dim, input_dims, num_trials, num_warmup, device_id
    )
    baseline_data = load_baseline_file(language, hardware, dims_key, torch_compile)
    if baseline_data and cache_key in baseline_data:
        entry = baseline_data[cache_key]
        if (
            isinstance(entry, dict)
            and "mean" in entry
            and not entry.get("not_supported")
        ):
            return entry
    return None


def save_cached_baseline(
    language: str,
    hardware: str,
    function: str,
    entry: Dict[str, Any],
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None,
    torch_compile: bool = False,
    num_trials: Optional[int] = None,
    num_warmup: Optional[int] = None,
    device_id: Optional[int] = None,
):
    dims_key, cache_key = _build_cache_key(
        function, batch_size, dim, input_dims, num_trials, num_warmup, device_id
    )
    baseline_data = (
        load_baseline_file(language, hardware, dims_key, torch_compile) or {}
    )
    baseline_data[cache_key] = entry
    save_baseline_file(language, hardware, baseline_data, dims_key, torch_compile)


def get_single_baseline(
    language: str,
    function: str,
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None,
    torch_compile: bool = False,
    num_trials: Optional[int] = None,
    num_warmup: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get baseline for a single function with optional custom dimensions.
    Returns dict with 'baseline', 'hardware', 'cached', 'function_code' keys.
    """
    backend = get_backend(language)
    hardware = backend.get_hardware_name()
    capability = backend.get_compute_capability()
    compute_capability = f"{capability[0]}.{capability[1]}" if capability else None

    # Read reference code
    function_code = _read_reference_code(function)

    dims_key, cache_key = _build_cache_key(
        function, batch_size, dim, input_dims, num_trials, num_warmup
    )

    cached = get_cached_baseline(
        language,
        function,
        hardware,
        batch_size=batch_size,
        dim=dim,
        input_dims=input_dims,
        torch_compile=torch_compile,
        num_trials=num_trials,
        num_warmup=num_warmup,
    )
    if cached:
        baseline = cached
        if (
            isinstance(baseline, dict)
            and "mean" in baseline
            and not baseline.get("not_supported")
        ):
            # Extract dimensions from cached baseline if available
            cached_batch_size = baseline.get("batch_size", batch_size)
            cached_dim = baseline.get("dim", dim)
            cached_input_dims = baseline.get("input_dims", input_dims)

            logger.info(
                f"[BASELINE] Function: {function}, Language: {language}, "
                f"Performance: {baseline['mean']:.3g}ms, Cached: True, Compiled: {torch_compile}"
            )

            return {
                "baseline": baseline,
                "hardware": hardware,
                "compute_capability": compute_capability,
                "cached": True,
                "torch_compile": torch_compile,
                "function_code": function_code,
                "batch_size": cached_batch_size,
                "dim": cached_dim,
                "input_dims": cached_input_dims,
            }

    # Compute baseline with custom dimensions
    try:
        baseline = compute_baseline(
            function,
            language,
            batch_size=batch_size,
            dim=dim,
            input_dims=input_dims,
            torch_compile=torch_compile,
            num_trials=num_trials,
            num_warmup=num_warmup,
        )
    except KeyError as e:
        # Function not found in dataset
        raise ValueError(f"Function '{function}' not found in dataset") from e
    except FileNotFoundError as e:
        # Reference file not found
        raise ValueError(
            f"Reference implementation not found for function '{function}'"
        ) from e

    if isinstance(baseline, dict) and baseline.get("not_supported"):
        error_msg = baseline.get("error", "Unknown error")
        error_type = baseline.get("error_type", "Unknown")
        raise ValueError(
            f"Baseline computation not supported for {function} on {language}: {error_type} - {error_msg}"
        )

    save_cached_baseline(
        language,
        hardware,
        function,
        baseline,
        batch_size=batch_size,
        dim=dim,
        input_dims=input_dims,
        torch_compile=torch_compile,
        num_trials=num_trials,
        num_warmup=num_warmup,
    )

    result = {
        "baseline": baseline,
        "hardware": hardware,
        "compute_capability": compute_capability,
        "cached": False,
        "torch_compile": torch_compile,
        "function_code": function_code,
        "batch_size": batch_size,
        "dim": dim,
        "input_dims": input_dims,
    }

    logger.info(
        f"[BASELINE] Function: {function}, Language: {language}, "
        f"Performance: {baseline['mean']:.3g}ms, Cached: False, Compiled: {torch_compile}"
    )

    return result


def get_all_baselines(language: str, torch_compile: bool = False) -> Dict[str, Any]:
    """
    Get all baselines for a language.
    Returns dict with 'baselines', 'hardware', 'cached', 'function_codes' keys.
    """
    backend = get_backend(language)
    hardware = backend.get_hardware_name()
    capability = backend.get_compute_capability()
    compute_capability = f"{capability[0]}.{capability[1]}" if capability else None

    # Try to load from file
    baseline_data = load_baseline_file(language, hardware, torch_compile=torch_compile)

    # Read reference codes for all functions
    function_codes = {}

    for func_name in dataset.keys():
        code = _read_reference_code(func_name)
        if code:
            function_codes[func_name] = code

    if baseline_data:
        valid_baselines = {
            k: v
            for k, v in baseline_data.items()
            if isinstance(v, dict) and "mean" in v and not v.get("not_supported")
        }
        if valid_baselines:
            return {
                "baselines": valid_baselines,
                "hardware": hardware,
                "compute_capability": compute_capability,
                "cached": True,
                "torch_compile": torch_compile,
                "function_codes": function_codes if function_codes else None,
            }

    baselines = compute_all_baselines(language, torch_compile=torch_compile)

    valid_baselines = {
        k: v
        for k, v in baselines.items()
        if isinstance(v, dict) and "mean" in v and not v.get("not_supported")
    }

    # Save to file
    save_baseline_file(language, hardware, valid_baselines, torch_compile=torch_compile)

    return {
        "baselines": valid_baselines,
        "hardware": hardware,
        "compute_capability": compute_capability,
        "cached": False,
        "torch_compile": torch_compile,
        "function_codes": function_codes if function_codes else None,
    }
