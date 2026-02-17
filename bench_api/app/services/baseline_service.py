import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from app.config import BASELINES_DIR, REFERENCE_DIR, MULTIKERNELBENCH_PATH
from app.core.bench_kernel import compute_baseline, compute_all_baselines, get_backend
from app.integration import get_reference_path

logger = logging.getLogger(__name__)

# Import dataset to get function category
sys.path.insert(0, str(MULTIKERNELBENCH_PATH))
from dataset import dataset


def get_baseline_file_path(language: str, hardware: str, dims_key: Optional[str] = None) -> Path:
    """Get the path to the baseline file for a language and hardware."""
    if dims_key:
        return BASELINES_DIR / f"{language}_{hardware}_{dims_key}.json"
    return BASELINES_DIR / f"{language}_{hardware}.json"


def _create_dims_key(batch_size: Optional[int] = None, dim: Optional[int] = None, input_dims: Optional[Dict[str, Any]] = None) -> Optional[str]:
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


def load_baseline_file(language: str, hardware: str, dims_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load baseline file if it exists."""
    baseline_path = get_baseline_file_path(language, hardware, dims_key)
    if baseline_path.exists():
        try:
            with open(baseline_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load baseline file {baseline_path}: {e}")
    return None


def save_baseline_file(language: str, hardware: str, data: Dict[str, Any], dims_key: Optional[str] = None):
    """Save baseline data to file."""
    baseline_path = get_baseline_file_path(language, hardware, dims_key)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, 'w') as f:
        json.dump(data, f, indent=2)


def _read_reference_code(function: str) -> Optional[str]:
    """Read reference code for a function."""
    try:
        ref_src_path = get_reference_path(function)
        if ref_src_path:
            with open(ref_src_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        logger.warning(f"Failed to read reference code for {function}: {e}")
    return None


def get_single_baseline(
    language: str, 
    function: str,
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None
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
    
    # Create dimensions key for cache
    dims_key = _create_dims_key(batch_size, dim, input_dims)
    
    # Try to load from file
    baseline_data = load_baseline_file(language, hardware, dims_key)
    
    # Create cache key for this function with dimensions
    cache_key = function
    if dims_key:
        cache_key = f"{function}_{dims_key}"
    
    if baseline_data and cache_key in baseline_data:
        baseline = baseline_data[cache_key]
        if isinstance(baseline, dict) and 'mean' in baseline and not baseline.get('not_supported'):
            # Extract dimensions from cached baseline if available
            cached_batch_size = baseline.get('batch_size', batch_size)
            cached_dim = baseline.get('dim', dim)
            cached_input_dims = baseline.get('input_dims', input_dims)
            
            logger.info(
                f"[BASELINE] Function: {function}, Language: {language}, "
                f"Performance: {baseline['mean']:.3g}ms, Cached: True"
            )
            
            return {
                'baseline': baseline,
                'hardware': hardware,
                'compute_capability': compute_capability,
                'cached': True,
                'function_code': function_code,
                'batch_size': cached_batch_size,
                'dim': cached_dim,
                'input_dims': cached_input_dims
            }
    
    # Compute baseline with custom dimensions
    try:
        baseline = compute_baseline(function, language, batch_size=batch_size, dim=dim, input_dims=input_dims)
    except KeyError as e:
        # Function not found in dataset
        raise ValueError(f"Function '{function}' not found in dataset") from e
    except FileNotFoundError as e:
        # Reference file not found
        raise ValueError(f"Reference implementation not found for function '{function}'") from e
    
    if isinstance(baseline, dict) and baseline.get('not_supported'):
        error_msg = baseline.get('error', 'Unknown error')
        error_type = baseline.get('error_type', 'Unknown')
        raise ValueError(f"Baseline computation not supported for {function} on {language}: {error_type} - {error_msg}")
    
    # Update or create baseline file
    if baseline_data is None:
        baseline_data = {}
    baseline_data[cache_key] = baseline
    save_baseline_file(language, hardware, baseline_data, dims_key)
    
    result = {
        'baseline': baseline,
        'hardware': hardware,
        'compute_capability': compute_capability,
        'cached': False,
        'function_code': function_code,
        'batch_size': batch_size,
        'dim': dim,
        'input_dims': input_dims
    }

    logger.info(
        f"[BASELINE] Function: {function}, Language: {language}, "
        f"Performance: {baseline['mean']:.3g}ms, Cached: False"
    )
    
    return result


def get_all_baselines(language: str) -> Dict[str, Any]:
    """
    Get all baselines for a language.
    Returns dict with 'baselines', 'hardware', 'cached', 'function_codes' keys.
    """
    backend = get_backend(language)
    hardware = backend.get_hardware_name()
    capability = backend.get_compute_capability()
    compute_capability = f"{capability[0]}.{capability[1]}" if capability else None
    
    # Try to load from file
    baseline_data = load_baseline_file(language, hardware)
    
    # Read reference codes for all functions
    function_codes = {}
    from dataset import dataset
    for func_name in dataset.keys():
        code = _read_reference_code(func_name)
        if code:
            function_codes[func_name] = code
    
    if baseline_data:
        valid_baselines = {
            k: v for k, v in baseline_data.items()
            if isinstance(v, dict) and 'mean' in v and not v.get('not_supported')
        }
        if valid_baselines:
            return {
                'baselines': valid_baselines,
                'hardware': hardware,
                'compute_capability': compute_capability,
                'cached': True,
                'function_codes': function_codes if function_codes else None
            }
    
    baselines = compute_all_baselines(language)
    
    valid_baselines = {
        k: v for k, v in baselines.items()
        if isinstance(v, dict) and 'mean' in v and not v.get('not_supported')
    }
    
    # Save to file
    save_baseline_file(language, hardware, valid_baselines)
    
    return {
        'baselines': valid_baselines,
        'hardware': hardware,
        'compute_capability': compute_capability,
        'cached': False,
        'function_codes': function_codes if function_codes else None
    }

