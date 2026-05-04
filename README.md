# LLM Low-Level Code Optimization — Evaluation Bundle

Evaluation infrastructure for the KernelBench benchmark: a FastAPI service that compiles submitted CUDA kernels, checks correctness, and measures latency against PyTorch reference implementations.

## Layout

| Path | Role |
|------|------|
| `KernelBench/` | Reference tasks: `level1`, `level2` (benchmark-native shapes) and `level1-small`, `level2-small` (small-input regime). |
| `bench_api/` | FastAPI service: compile, correctness check, latency measurement vs. references. |

## Quick start

```bash
# 1. Create and activate environment
conda create -n bench_api python=3.10 -y
conda activate bench_api

# 2. Install dependencies + PyTorch (adjust index URL for your CUDA version)
cd bench_api
pip install -r requirements.txt
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Start server (tasks are discovered from ../KernelBench automatically)
./run.sh
```

The API is available at `http://localhost:8123`. Interactive docs: `http://localhost:8123/docs`.

Optional experiment logging: set `ENABLE_MLFLOW=true` and `MLFLOW_TRACKING_URI=...`
(MLflow is **off** by default).

Full documentation: `bench_api/README.md`.
