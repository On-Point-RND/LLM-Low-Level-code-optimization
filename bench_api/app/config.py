import os
import logging
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
MULTIKERNELBENCH_PATH = Path(os.getenv("MULTIKERNELBENCH_PATH", str(BASE_DIR / "MultiKernelBench")))
KERNELBENCH_PATH = Path(os.getenv("KERNELBENCH_PATH", str(BASE_DIR / "KernelBench/KernelBench")))
BASELINES_DIR = Path(os.getenv("BASELINES_DIR", str(MULTIKERNELBENCH_PATH / "baselines")))
REFERENCE_DIR = Path(os.getenv("REFERENCE_DIR", str(MULTIKERNELBENCH_PATH / "reference")))

DEFAULT_HARDWARE = os.getenv("DEFAULT_HARDWARE", None)  

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

log_file = LOGS_DIR / f"bench_api_{datetime.now().strftime('%Y%m%d')}.log"

root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

console_handler = logging.StreamHandler()
console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
console_handler.setFormatter(logging.Formatter(log_format, date_format))

file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
file_handler.setFormatter(logging.Formatter(log_format, date_format))

if not root_logger.handlers:
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

ENABLE_TORCH_COMPILE = os.getenv("ENABLE_TORCH_COMPILE", "false").lower() == "true"

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://195.209.214.105:5050")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "MultiKernelBench")
ENABLE_MLFLOW = os.getenv("ENABLE_MLFLOW", "true").lower() == "true"
