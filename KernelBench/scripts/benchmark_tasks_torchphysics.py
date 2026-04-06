#!/usr/bin/env python3
"""
benchmark_tasks_torchphysics.py
================================
Runs every task in  tasks/sci_bench_torchphysics/  and prints a table comparing
the embedded pure-PyTorch PDE residual solver against torchphysics operators.

Tasks are discovered recursively across all subdirectories:
  tasks/sci_bench_torchphysics/
    level1/             # L1 operator primitives (laplacian, grad, div, …)
    first_order/        # PDE residuals: gradient / divergence operators
    second_order/       # PDE residuals: Laplacian-based operators
    higher_order/       # PDE residuals: biharmonic / 4th-order operators
    nonlinear_composition/  # PDE residuals: nonlinear operator compositions

For each task:
  • Instantiates Model, calls forward(*inputs).
  • Calls make_torchphysics_ref(model) and runs it with the same inputs.
  • Reports max-absolute error, relative error, and wall-clock timing.

Because both implementations use the same network weights and the same
collocation points, and both ultimately call torch.autograd.grad in the same
order, outputs should match to floating-point precision (rel-err ≈ 0).

Note: All tasks use autograd internally.  forward() wraps its body in
  torch.enable_grad(), so timing runs correctly even inside torch.no_grad().

Usage
-----
    python scripts/benchmark_tasks_torchphysics.py                        # all tasks
    python scripts/benchmark_tasks_torchphysics.py --no-timing            # correctness only
    python scripts/benchmark_tasks_torchphysics.py --filter Poisson
    python scripts/benchmark_tasks_torchphysics.py --subdir level1        # one subfolder
    python scripts/benchmark_tasks_torchphysics.py --subdir second_order
    python scripts/benchmark_tasks_torchphysics.py --n-trials 20
    python scripts/benchmark_tasks_torchphysics.py --device cpu
"""

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT      = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "ScientificKernelBenchFunctional"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_task(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _timeit(fn, device: str, n_warmup: int, n_trials: int) -> float:
    # Note: forward() uses torch.enable_grad() internally, so no_grad is fine here
    with torch.no_grad():
        for _ in range(n_warmup):
            fn()
    _sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_trials):
            fn()
    _sync(device)
    return (time.perf_counter() - t0) / n_trials


# ── Per-task runner ───────────────────────────────────────────────────────────

def run_task(path: Path, device: str,
             n_warmup: int, n_trials: int, do_timing: bool,
             err_threshold: float) -> dict:
    name = path.stem
    rec  = dict(name=name, error=None, rel_error=None,
                t_ours=None, t_ref=None, status="ok", note="")

    mod    = _load_task(path)
    inputs = [t.to(device) for t in mod.get_inputs()]
    model  = mod.Model().to(device).eval()

    # ── Our pure-PyTorch forward ──────────────────────────────────────────────
    with torch.no_grad():
        out_ours = model(*inputs)

    # ── torchphysics reference ────────────────────────────────────────────────
    if not hasattr(mod, "make_torchphysics_ref"):
        rec["status"] = "no-ref"
        rec["note"]   = "make_torchphysics_ref not defined"
        return rec

    try:
        ref_layer = mod.make_torchphysics_ref(model).to(device).eval()
    except Exception as exc:
        rec["status"] = "ref-error"
        rec["note"]   = str(exc)[:80]
        return rec

    try:
        with torch.no_grad():
            out_ref = ref_layer(*inputs)
    except Exception as exc:
        rec["status"] = "ref-error"
        rec["note"]   = str(exc)[:80]
        return rec

    # ── Error ─────────────────────────────────────────────────────────────────
    abs_err = (out_ours.double() - out_ref.double()).abs().max()
    scale   = out_ref.double().abs().max().clamp(min=1.0)
    rec["error"]     = abs_err.item()
    rec["rel_error"] = (abs_err / scale).item()

    if rec["rel_error"] > err_threshold:
        rec["status"] = "FAIL"
        rec["note"]   = f"rel={rec['rel_error']:.2e} > {err_threshold:.0e}"

    # ── Timing ────────────────────────────────────────────────────────────────
    if do_timing and rec["status"] == "ok":
        rec["t_ours"] = _timeit(lambda: model(*inputs),
                                device, n_warmup, n_trials)
        try:
            rec["t_ref"] = _timeit(lambda: ref_layer(*inputs),
                                   device, n_warmup, n_trials)
        except Exception:
            pass

    return rec


# ── Table ─────────────────────────────────────────────────────────────────────

def _fmt_err(v):
    if v is None: return "     N/A"
    if v == 0.0:  return "  0 (exact)"
    return f"{v:.3e}"

def _fmt_ms(v):
    return f"{v * 1000:10.3f}" if v is not None else "       N/A"

def _fmt_spd(t_ours, t_ref):
    if t_ours is None or t_ref is None or t_ours == 0:
        return "    N/A"
    ratio = t_ref / t_ours
    tag   = "faster" if ratio >= 1.0 else "slower"
    return f"{ratio:5.2f}x ({tag})"


def print_table(rows: list, do_timing: bool):
    W = max(len(r["name"]) for r in rows)

    if do_timing:
        hdr = (f"{'Task':<{W}}  {'Max|err|':>11}  {'Rel|err|':>10}  "
               f"{'PyTorch(ms)':>11}  {'torchphysics(ms)':>16}  "
               f"{'Speedup':>14}  Status")
    else:
        hdr = f"{'Task':<{W}}  {'Max|err|':>11}  {'Rel|err|':>10}  Status"

    sep = "─" * len(hdr)
    print(sep); print(hdr); print(sep)

    prev_op = None
    for r in rows:
        # Group by problem prefix (strip trailing _small/_medium/_large)
        parts = r["name"].split("_")
        op = "_".join(parts[1:-1]) if len(parts) > 2 else r["name"]
        if op != prev_op:
            if prev_op is not None:
                print()
            prev_op = op

        err_s  = _fmt_err(r["error"])
        rel_s  = _fmt_err(r.get("rel_error"))
        status = r["status"]
        note   = f"  [{r['note']}]" if r["note"] else ""

        if do_timing:
            spd  = _fmt_spd(r["t_ours"], r["t_ref"])
            line = (f"{r['name']:<{W}}  {err_s:>11}  {rel_s:>10}  "
                    f"{_fmt_ms(r['t_ours']):>11}  {_fmt_ms(r['t_ref']):>16}  "
                    f"{spd:>14}  {status}{note}")
        else:
            line = f"{r['name']:<{W}}  {err_s:>11}  {rel_s:>10}  {status}{note}"
        print(line)

    print(sep)

    ok      = [r for r in rows if r["status"] == "ok"]
    failed  = [r for r in rows if r["status"] == "FAIL"]
    errored = [r for r in rows if r["status"] in ("ref-error", "ERROR")]
    no_ref  = [r for r in rows if r["status"] == "no-ref"]

    if ok:
        rels = [r["rel_error"] for r in ok if r.get("rel_error") is not None]
        print(f"\n  {len(ok)}/{len(rows)} tasks OK"
              + (f"  |  median rel-err = {sorted(rels)[len(rels)//2]:.2e}"
                 f"  |  max rel-err = {max(rels):.2e}" if rels else ""))
    if failed:
        print(f"  {len(failed)} FAILED (error above threshold)")
    if errored:
        print(f"  {len(errored)} ERRORED (exception)")
    if no_ref:
        print(f"  {len(no_ref)} skipped (no make_torchphysics_ref)")

    if do_timing and ok:
        spds = [r["t_ref"] / r["t_ours"]
                for r in ok if r["t_ours"] and r["t_ref"] and r["t_ours"] > 0]
        if spds:
            print(f"\n  Speedup (pure-PyTorch vs torchphysics):  "
                  f"median {sorted(spds)[len(spds)//2]:.2f}x  |  "
                  f"min {min(spds):.2f}x  |  max {max(spds):.2f}x")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--filter",        default="",
                        help="Substring filter on task filename")
    parser.add_argument("--subdir",        default="",
                        help="Restrict to a specific subdirectory "
                             "(e.g. level1, first_order, second_order, "
                             "higher_order, nonlinear_composition)")
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-warmup",      type=int, default=3)
    parser.add_argument("--n-trials",      type=int, default=10)
    parser.add_argument("--no-timing",     action="store_true")
    parser.add_argument("--err-threshold", type=float, default=1e-5,
                        help="Max relative error to pass (default 1e-5)")
    args = parser.parse_args()

    search_root = TASKS_DIR / args.subdir if args.subdir else TASKS_DIR
    if not search_root.is_dir():
        sys.exit(f"Subdirectory not found: {search_root}")

    task_files = sorted(search_root.rglob("[0-9]*.py"))
    if args.filter:
        task_files = [f for f in task_files if args.filter in f.name]
    if not task_files:
        sys.exit(f"No task files found in {search_root} matching {args.filter!r}")

    do_timing = not args.no_timing
    device    = args.device

    print(f"\nRunning {len(task_files)} tasks on {device}"
          + ("  (timing enabled)" if do_timing else "  (correctness only)")
          + (f"  [subdir: {args.subdir}]" if args.subdir else "")
          + "\n")

    rows = []
    for path in task_files:
        # Use subdir/stem as the display name so the table is self-explanatory
        rel_path = path.relative_to(TASKS_DIR)
        display_name = str(rel_path.parent / path.stem) if rel_path.parent != Path(".") else path.stem
        try:
            rec = run_task(path, device, args.n_warmup, args.n_trials,
                           do_timing, args.err_threshold)
        except Exception as exc:
            rec = dict(name=display_name, error=None, rel_error=None,
                       t_ours=None, t_ref=None,
                       status="ERROR", note=str(exc)[:80])
            traceback.print_exc()
        rec["name"] = display_name
        rows.append(rec)
        rel = rec.get("rel_error")
        err_s = f"{rel:.2e}" if rel is not None else "N/A"
        print(f"  {display_name:70s}  rel={err_s:>10}  {rec['status']}")

    print()
    print_table(rows, do_timing)


if __name__ == "__main__":
    main()
