import logging
from pathlib import Path
from typing import Optional

from app.config import BENCH_DIRS

logger = logging.getLogger(__name__)

# Unified dataset: func_name -> {"category": str, "ref_path": str, "bench_dir": str}
_dataset: dict[str, dict] = {}
_dataset_loaded = False


def _get_reference_root(bench_dir: Path) -> Path:
    """Return the root that contains category subdirs for a bench dir."""
    reference_subdir = bench_dir / "reference"
    if reference_subdir.is_dir():
        return reference_subdir
    return bench_dir


def _register_bench_dir(bench_dir: Path) -> int:
    """Scan a benchmark directory and register all functions into _dataset."""
    if not bench_dir.exists():
        logger.warning(f"Bench dir does not exist, skipping: {bench_dir}")
        return 0

    ref_root = _get_reference_root(bench_dir)
    count = 0
    for category_dir in sorted(ref_root.iterdir()):
        if not category_dir.is_dir():
            continue
        for file in sorted(category_dir.glob("*.py")):
            func_name = file.stem
            if func_name not in _dataset:
                _dataset[func_name] = {
                    "category": category_dir.name,
                    "ref_path": str(file),
                    "bench_dir": str(bench_dir),
                }
                count += 1

    return count


def _ensure_loaded():
    global _dataset_loaded
    if _dataset_loaded:
        return
    _dataset_loaded = True

    for bench_dir in BENCH_DIRS:
        count = _register_bench_dir(bench_dir)
        logger.info(f"Registered {count} functions from {bench_dir}")


def register_all_bench_dirs():
    """Load all benchmark directories listed in BENCH_DIRS."""
    global _dataset_loaded
    _dataset_loaded = False  # force reload
    _dataset.clear()
    _ensure_loaded()


def get_dataset() -> dict:
    _ensure_loaded()
    return _dataset


def get_reference_path(function: str) -> Optional[str]:
    _ensure_loaded()
    info = _dataset.get(function)
    if info is None:
        return None
    ref_path = info.get("ref_path")
    if ref_path and Path(ref_path).exists():
        return ref_path
    return None
