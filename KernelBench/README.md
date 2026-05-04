# KernelBench task snippets (subset)

Each `.py` file is one reference PyTorch module + `get_inputs()` used by the evaluator.

- `level1/`, `level2/` — benchmark-native tensor shapes.  
- `level1-small/`, `level2-small/` — matched small-input regime (distribution shift in the paper).

The API discovers tasks from these directories via `BENCH_DIRS` (see `bench_api/app/config.py`).
