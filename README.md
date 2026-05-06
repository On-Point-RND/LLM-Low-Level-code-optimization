# LLM Low-Level Code Optimization — Evaluation Bundle

This repository bundles **KernelBench-style reference tasks**, a **FastAPI evaluation service** for submitted CUDA, and **two optional agent pipelines**: one that searches **raw CUDA** via the API, and one that optimizes **TVM Relax / TIR** with LLM-assisted transforms. Together they support experiments that compare PyTorch references, hand-written or generated CUDA, and TVM-compiled stacks.

## Repository layout

| Path | Role |
|------|------|
| **`KernelBench/`** | Reference PyTorch modules (`Model`, `get_inputs`, …) grouped by difficulty / regime: `level1`, `level2`, `level1-small`, `level2-small`. Consumed by **`bench_api`** (and can be pointed at by **`tvm_agent`** via `kernelbench_root` in `tvm_agent/config.yaml`). |
| **`bench_api/`** | FastAPI service: discover tasks from `KernelBench`, compile submitted CUDA, check correctness against references, measure latency. Optional MLflow logging. Defaults to `http://localhost:8123`. |
| **`cuda_agent/`** | LLM-driven CUDA kernel search: proposes candidates, sends them to **`bench_api`**, logs runs (e.g. Parquet under `data/<exp-id>/`). Configure models and search strategies in `configs.toml`. Expects `BENCHAPI_URL` if the API is not on localhost. |
| **`tvm_agent/`** | Offline batch runner: PyTorch → ONNX / Relax → MetaSchedule (optional) → **multi-agent LLM loop** on TIR (analysis / code / summary), then build, correctness, CUDA profiling. Writes `res/` per task. Requires a local TVM build (`tvm/python` symlink or equivalent) and OpenAI-compatible endpoints in `config.yaml`. |
| **`.gitignore`** | Ignores caches, logs, MLflow artifacts, temp dirs, and OS junk at repo root. |

There is **no single “run everything” command**: start **`bench_api`** when you use **`cuda_agent`**; run **`tvm_agent`** separately when you evaluate the TVM stack (point it at this repo’s **`KernelBench/`** if you do not keep a copy under `tvm_agent/KernelBench`).

## How the pieces connect

- **`KernelBench/`** is the shared task source for references and input generators.
- **`bench_api`** is the **gatekeeper for CUDA**: compilation, correctness, timing vs PyTorch.
- **`cuda_agent`** is a **client of `bench_api`** for iterative or tree-style LLM search over CUDA code.
- **`tvm_agent`** is **independent of `bench_api`**: it drives TVM internally and only shares the same *idea* of tasks as `KernelBench` (layout-compatible `.py` files).

