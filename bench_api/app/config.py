import os
import logging
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
MULTIKERNELBENCH_PATH = Path(
    os.getenv("MULTIKERNELBENCH_PATH", str(BASE_DIR.parent / "MultiKernelBench"))
)
KERNELBENCH_PATH = Path(
    os.getenv("KERNELBENCH_PATH", str(BASE_DIR.parent / "KernelBench/KernelBench"))
)
BASELINES_DIR = Path(os.getenv("BASELINES_DIR", str(BASE_DIR / "baselines")))
CUDA_BUILD_ROOT = Path(os.getenv("CUDA_BUILD_ROOT", "/tmp/bench_builds"))
REFERENCE_DIR = Path(
    os.getenv("REFERENCE_DIR", str(MULTIKERNELBENCH_PATH / "reference"))
)

DEFAULT_HARDWARE = os.getenv("DEFAULT_HARDWARE", None)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

log_file = LOGS_DIR / f"bench_api_{datetime.now().strftime('%Y%m%d')}.log"

root_logger = logging.getLogger()
log_level_numeric = getattr(logging, LOG_LEVEL, logging.INFO)
root_logger.setLevel(log_level_numeric)

# Update existing handlers level if they exist (e.g. from uvicorn)
for handler in root_logger.handlers:
    handler.setLevel(log_level_numeric)

console_handler = logging.StreamHandler()
console_handler.setLevel(log_level_numeric)
console_handler.setFormatter(logging.Formatter(log_format, date_format))

file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(log_level_numeric)
file_handler.setFormatter(logging.Formatter(log_format, date_format))

# Only add our handlers if they are not already there
# We check by type to avoid duplicate console/file handlers
has_console = any(
    isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    for h in root_logger.handlers
)
has_file = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)

if not has_console:
    root_logger.addHandler(console_handler)
if not has_file:
    root_logger.addHandler(file_handler)

ENABLE_TORCH_COMPILE = os.getenv("ENABLE_TORCH_COMPILE", "false").lower() == "true"

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://195.209.214.105:5050")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "MultiKernelBench")
ENABLE_MLFLOW = os.getenv("ENABLE_MLFLOW", "true").lower() == "true"

# Execution Configuration
NUM_CORRECT_TRIALS = int(os.getenv("NUM_CORRECT_TRIALS", "5"))
NUM_PERF_TRIALS = int(os.getenv("NUM_PERF_TRIALS", "100"))
NUM_WARMUP = int(os.getenv("NUM_WARMUP", "3"))
SEED_NUM = int(os.getenv("SEED_NUM", "1024"))
ARCH_LIST = os.getenv("ARCH_LIST", None)
if ARCH_LIST:
    ARCH_LIST = ARCH_LIST.split(",")

# Comma-separated CUDA device indices to use for evaluation, e.g. "0,1,2,3"
DEVICE_IDS: list[int] = [
    int(x.strip()) for x in os.getenv("DEVICE_IDS", "0").split(",") if x.strip()
]

# Comma-separated backends to enable at startup, e.g. "cuda,triton"
BACKENDS: list[str] = [
    x.strip() for x in os.getenv("BACKENDS", "cuda").split(",") if x.strip()
]

# Comma-separated list of benchmark directories to load at startup.
# Each dir should contain category subdirs with .py reference files, either:
#   - directly: {bench_dir}/{category}/{func}.py  (KernelBench-style)
#   - or under reference/: {bench_dir}/reference/{category}/{func}.py  (MultiKernelBench-style)
# Defaults to MultiKernelBench and KernelBench paths if not set.
_default_bench_dirs = ",".join(
    str(p)
    for p in [
        BASE_DIR.parent / "MultiKernelBench",
        BASE_DIR.parent / "KernelBench" / "KernelBench",
    ]
)
BENCH_DIRS: list[Path] = [
    Path(p.strip())
    for p in os.getenv("BENCH_DIRS", _default_bench_dirs).split(",")
    if p.strip()
]
