# kernel-agent

LLM-driven CUDA kernel search over KernelBench-style tasks: propose kernels, compile and evaluate them via a local **bench_api**, and log every candidate to Parquet.

## Prerequisites

- **Python** ≥ 3.13 (see `pyproject.toml`).
- **OpenAI-compatible inference** (e.g. vLLM) reachable at the `base_url` you set in `configs.toml`.
- **bench_api** running for compile/correctness/speed evaluation (default `http://localhost:8123`). Override with:

  ```bash
  export BENCHAPI_URL=http://host:port
  ```

- If a model block omits `api_key`, the client falls back to **`OPENAI_API_KEY`**.

Optional:

```bash
export KERNEL_AGENT_SKIP_HARDWARE_PROBE=1   # skip POST /baseline at startup
```

## Install

From this directory:

```bash
uv sync
# or
pip install -e .
```

## Run an experiment

```bash
python main.py \
  --exp-id my-exp \
  --model qwen3-coder-next \
  --search-method iterative \
  --dataset kernelbench
```

`--dataset` supports `kernelbench` or `multikernelbench`.

**Filters**

| Flag | Meaning |
|------|--------|
| `--categories level1,level2` | Restrict benchmark categories |
| `--kernels add,relu` | Restrict kernels by name (comma-separated stems) |
| `--parallel-kernels N` | Run up to `N` kernels concurrently (default `1`) |

**Resume:** results go to `data/<exp-id>/results.parquet`. Re-running the same `--exp-id` continues from existing rows.

## Configuration (`configs.toml`)

**Models** — one TOML table per `--model` name, with `model_name`, `base_url`, and optionally `api_key`.

**Search methods** — tables such as `iterative`, `sampling`, `tree-search`. Conceptually:

| `width` × `height` | Behaviour |
|--------------------|-----------|
| `1 × N` | Iterative refinement |
| `N × 1` | Parallel sampling |
| `W × H` | Tree / beam-style expansion |

Shared sampling knobs: `temperature`, `top_p`, `beam_size`, `two_stage`, etc. Per-stage caps live under `[stage]` (`diagnoser_*`, `advisor_*`, `coder_max_tokens`).

Add a new method block, then pass `--search-method <table-name>`.

## Output layout

```
data/
└── <exp-id>/
    └── results.parquet
```

Each row is one generated candidate. Useful columns:

| Column | Description |
|--------|-------------|
| `model_name` | Resolved model id |
| `search_method` | Config name |
| `iteration` | Depth index (0-based) |
| `kernel_category` | e.g. `level1` |
| `kernel_name` | Task stem |
| `dataset` | `kernelbench` / `multikernelbench` |
| `compiled` / `correctness` | Compile and numerical match |
| `speedup` | vs reference (if correct) |
| `generated_code` / `reference_code` | Full sources |
| `hypothesis` | Model’s stated optimization idea |
| `node_id` / `parent_node_id` | Tree linkage |
| `input_tokens` / `output_tokens` / `cached_tokens` | Usage |
| `generation_timing` / `evaluation_timing_total` | Seconds |

