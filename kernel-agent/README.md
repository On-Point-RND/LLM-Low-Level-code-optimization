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
width       = 3    # candidates per node
height      = 4    # tree depth (iterations)
beam_size   = 2    # nodes kept between levels
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

Prints per-category and per-model tables with `Comp@k`, `Pass@k`, `SU1@k` (speedup > 1×), `AvgSU@k`, and `MaxSU@k` for each iteration depth k.
