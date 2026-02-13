from config import project_root_path, ref_impl_base_path
from dataset import dataset
from backends.backend_registry import BACKEND_REGISTRY
from utils.utils import get_ref_src_path
import importlib
import os
import json
import multiprocessing as mp
import os
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Run baseline evaluation with specified parameters.")
parser.add_argument('--language', type=str, default='cuda', help='Language to use.')
args = parser.parse_args()
language = args.language


result = {}
op_tests = dataset.keys()

# This function will be run in a separate process
def run_op(op, return_dict):
    try:
        if language not in BACKEND_REGISTRY:
            try:
                importlib.import_module(f"backends.{language}_backend")
            except ImportError as e:
                raise ValueError(f"Unsupported language/platform: {language} (module not found)") from e
            backend = BACKEND_REGISTRY.get(language)
            if backend is None:
                raise ValueError(f"Unsupported language/platform: {language}")
        with open(get_ref_src_path(op), 'r') as f:
            ref_src = f.read()
            exec(ref_src, backend.context)
        elapsed_times = backend.time_execution('Model')
        return_dict[op] = {
            "mean": float(f"{np.mean(elapsed_times):.3g}"),
            "std": float(f"{np.std(elapsed_times):.3g}"),
            "min": float(f"{np.min(elapsed_times):.3g}"),
            "max": float(f"{np.max(elapsed_times):.3g}"),
            "num_trials": len(elapsed_times),
            'device': backend.get_hardware_name()
        }

    except Exception as e:
        print(f"[ERROR] {op}: {e}")
        return_dict[op] = "not supported"

# Run each op in a separate process
for op in op_tests:
    print('[INFO]', op)
    manager = mp.Manager()
    return_dict = manager.dict()

    p = mp.Process(target=run_op, args=(op, return_dict))
    p.start()
    p.join(timeout=120)  # Timeout in seconds; adjust if needed

    if p.is_alive():
        p.terminate()
        print(f"[TIMEOUT] {op}")
        result[op] = "timeout"
    else:
        result[op] = return_dict.get(op, "not supported")
    device_name = result[op]['device']

os.makedirs(os.path.join(project_root_path, 'baselines'), exist_ok=True)

print(f'[INFO] Write to baselines/{language}_{device_name}.json')
with open(os.path.join(project_root_path, 'baselines', f'{language}_{device_name}.json'), 'w') as f:
    json.dump(result, f, indent=2)
