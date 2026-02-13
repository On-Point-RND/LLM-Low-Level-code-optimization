import sys
from pathlib import Path
from typing import Dict, Any

from app.config import MULTIKERNELBENCH_PATH

# Add MultiKernelBench to Python path
sys.path.insert(0, str(MULTIKERNELBENCH_PATH))

from dataset import dataset, category2exampleop
from backends.backend_registry import BACKEND_REGISTRY


def get_help_info() -> Dict[str, Any]:
    """
    Get help information about available languages, functions, and categories.
    
    Returns:
        Dictionary with help information
    """
    # Get available languages from backends
    # Try to import all known backends
    known_backends = ['cuda', 'triton', 'ascendc', 'sycl', 'pallas', 'tilelang_ascend']
    available_languages = []
    
    for lang in known_backends:
        try:
            if lang in BACKEND_REGISTRY:
                available_languages.append(lang)
            else:
                # Try to import
                import importlib
                importlib.import_module(f"backends.{lang}_backend")
                if lang in BACKEND_REGISTRY:
                    available_languages.append(lang)
        except ImportError:
            pass
    
    # Get functions and categories from dataset
    functions = {}
    categories = set()
    
    for func_name, func_data in dataset.items():
        category = func_data.get('category', 'unknown')
        functions[func_name] = category
        categories.add(category)
    
    # Endpoint descriptions
    endpoints = {
        "GET /help": {
            "description": "Get information about available languages, functions, and categories",
            "parameters": None
        },
        "POST /baseline": {
            "description": "Get baseline performance for a function or all functions for a language",
            "parameters": {
                "language": "Language/backend (cuda, triton, ascendc, sycl, pallas, tilelang_ascend)",
                "function": "Function name from dataset or 'all' to get all baselines"
            }
        },
        "POST /evaluate": {
            "description": "Evaluate kernel code performance",
            "parameters": {
                "torch_compile": "Whether to use torch.compile (bool, default: false)",
                "language": "Language/backend",
                "function": "Function name from dataset",
                "function_code": "Kernel code content (string, optional)",
                "function_code_file": "Kernel code as base64 or file path (string, optional)",
                "experiment_name": "MLflow experiment name - folder name in MLflow (string, optional)",
                "run_name": "MLflow run name - name of specific run (string, optional)",
                "num_trials": "Number of performance measurement trials (int, optional, default: 100)"
            }
        }
    }
    
    return {
        "languages": available_languages,
        "functions": functions,
        "categories": sorted(list(categories)),
        "endpoints": endpoints
    }

