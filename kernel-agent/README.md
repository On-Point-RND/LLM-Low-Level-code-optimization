# Kernel Agent

## Run an experiment

```bash
python main.py \
  --exp-id my-exp \
  --model qwen3-coder-next \
  --search-method iterative \
  --dataset kernelbench          # or: multikernelbench
```

Optional filters:
- `--categories level1,level2` — run only specific categories
- `--kernels add,relu` — run only specific kernels

Results are saved to `data/<exp-id>/results.parquet`. Re-running resumes from where it left off.

## Add a model

Add a section to `configs.toml`:

```toml
[my-model]
model_name = "MyModel"
base_url   = "http://localhost:8000/v1"
api_key    = "EMPTY"
```

Then pass `--model my-model` to `main.py`.

## Add a search method

```toml
[my-search]
width       = 3    # expansion candidates per node
height      = 4    # tree depth (refinement iterations)
beam_size   = 2    # nodes kept between levels (greedy search only)
temperature = 0.7
top_p       = 0.95
```

- `width=1, height=N` — iterative refinement
- `width=N, height=1` — parallel sampling
- `width=W, height=H` — beam/tree search

Then pass `--search-method my-search` to `main.py`.

## View results

```bash
python report.py --exp-id my-exp
```

Prints per-category and per-model tables with `Comp@k`, `Pass@k`, `SU1@k` (speedup > 1× and valid), `AvgSU@k` (geomean), and `MaxSU@k` for each iteration depth k.

## Data directory

```
data/
└── <exp-id>/
    └── results.parquet
```

Each row is one generated candidate. Key columns:

| Column | Description |
|---|---|
| `model_name` | Model identifier |
| `search_method` | Search method name |
| `iteration` | Depth index (0-based) |
| `kernel_category` | Problem category (e.g. `level1`) |
| `kernel_name` | Problem filename stem |
| `dataset` | `kernelbench` or `multikernelbench` |
| `compiled` | Whether the code compiled |
| `correctness` | Whether output matches reference |
| `speedup` | Speedup over reference (null if incorrect) |
| `generated_code` | Full generated kernel source |
| `reference_code` | Reference PyTorch implementation |
| `hypothesis` | Model's stated optimization hypothesis |
| `node_id` / `parent_node_id` | Tree node linkage |
| `input_tokens` / `output_tokens` / `cached_tokens` | Token usage |
| `generation_timing` | Generation time (s) |
| `evaluation_timing_total` | Evaluation time (s) |
