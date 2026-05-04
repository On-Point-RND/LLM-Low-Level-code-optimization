# KernelBench Evaluation API

FastAPI service for compiling, correctness-checking, and benchmarking CUDA kernels against PyTorch reference implementations from the KernelBench task suite.

---

## Quick start

```bash
# 1. Create environment
conda create -n bench_api python=3.10 -y
conda activate bench_api

# 2. Install dependencies (install PyTorch separately for your CUDA / CPU)
pip install -r requirements.txt
# CUDA 12.1:
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 --index-url https://download.pytorch.org/whl/cu121
# CPU only:
# pip install torch==2.2.1

# 3. Start the server (tasks load from ../KernelBench by default)
cd bench_api
./run.sh
```

The server listens on `http://localhost:8123` by default. Interactive API docs: `http://localhost:8123/docs`

Smoke check (no GPU required):

```bash
curl http://localhost:8123/help
```

---

## Installation

### Option 1: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate bench_api

# PyTorch — pick one:
# CUDA 12.1:
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 --index-url https://download.pytorch.org/whl/cu121
# CPU only:
# pip install torch==2.2.1
```

#### Low disk space

```bash
conda create -n bench_api python=3.10 -y
conda activate bench_api
./install_with_temp.sh
```

### Option 2: pip only

```bash
pip install -r requirements.txt
pip install torch==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

## Running the server

### Option 1: Bash script (recommended)

```bash
./run.sh
```

With environment variables:

```bash
HOST=0.0.0.0 PORT=8123 WORKERS=1 LOG_LEVEL=info ./run.sh
```

### Option 2: Python

```bash
python3 start_server.py --host 0.0.0.0 --port 8123
```

### Option 3: uvicorn

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8123
```

## API endpoints

### GET /help

Returns available backends, task names, and categories.

**Example response:**

```json
{
  "languages": ["cuda"],
  "functions": { "leaky_relu": "activation", "...": "..." },
  "categories": ["activation", "matmul", "..."],
  "endpoints": {}
}
```

### POST /baseline

Baseline latency for one task or all tasks for a backend.

**Request:**

```json
{
  "language": "cuda",
  "function": "leaky_relu"
}
```

**Response:**

```json
{
  "function": "leaky_relu",
  "language": "cuda",
  "hardware": "NVIDIA A100-SXM4-80GB",
  "baseline": { "mean": 0.123, "std": 0.001, "min": 0.120, "max": 0.125, "num_trials": 100 },
  "cached": true
}
```

### POST /evaluate

Evaluate submitted kernel code.

**Request:**

```json
{
  "torch_compile": false,
  "language": "cuda",
  "function": "leaky_relu",
  "function_code": "...",
  "experiment_name": "my_experiments",
  "run_name": "leaky_relu_run_001"
}
```

**Response:**

```json
{
  "function": "leaky_relu",
  "language": "cuda",
  "hardware": "NVIDIA A100-SXM4-80GB",
  "compiled": true,
  "correctness": true,
  "performance": { "mean": 0.115, "std": 0.001, "min": 0.113, "max": 0.118, "num_trials": 100 },
  "compile_info": null,
  "correctness_info": null,
  "run_name": "leaky_relu_run_001"
}
```

## Configuration

Environment variables:

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_DIRS` | `../KernelBench` | Comma-separated paths to task roots |
| `BASELINES_DIR` | `bench_api/baselines` | Baseline cache directory |
| `DEVICE_IDS` | `0` | CUDA device indices |
| `NUM_PERF_TRIALS` | `100` | Performance measurement iterations |
| `LOG_LEVEL` | `INFO` | Logging level |

### MLflow (disabled by default)

MLflow is **off** by default. To enable:

```bash
ENABLE_MLFLOW=true MLFLOW_TRACKING_URI=http://your-mlflow:5000 ./run.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_MLFLOW` | `false` | Enable MLflow logging |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | MLflow server URI |
| `MLFLOW_EXPERIMENT_NAME` | `KernelBench` | Default experiment name |

## Project layout

```
LLM-Low-Level-code-optimization/   # bundle root
├── KernelBench/                   # tasks (level1, level1-small, level2, level2-small)
│   └── README.md
├── bench_api/                     # this service
│   ├── app/
│   │   ├── main.py                # FastAPI app
│   │   ├── config.py              # settings
│   │   ├── models.py              # Pydantic models
│   │   ├── integration.py         # load tasks from BENCH_DIRS
│   │   ├── services/
│   │   └── core/                  # compile, benchmark, backends
│   ├── baselines/                 # baseline cache (JSON)
│   ├── run.sh
│   ├── start_server.py
│   ├── example_relu_cuda.py       # example CUDA submission
│   ├── setup_conda.sh
│   ├── install_with_temp.sh
│   ├── environment.yml
│   ├── requirements.txt
│   └── README.md
└── README.md                      # bundle quick start
```

## Examples

```bash
# List tasks
curl http://localhost:8123/help

# Baseline for one task
curl -X POST http://localhost:8123/baseline \
  -H "Content-Type: application/json" \
  -d '{"language": "cuda", "function": "leaky_relu"}'

# Evaluate a custom kernel
curl -X POST http://localhost:8123/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "language": "cuda",
    "function": "leaky_relu",
    "function_code": "import torch\nimport torch.nn as nn\n\nclass ModelNew(nn.Module):\n    def __init__(self): super().__init__()\n    def forward(self, x): return torch.nn.functional.leaky_relu(x, 0.01)"
  }'
```

Full CUDA example: `example_relu_cuda.py`.
