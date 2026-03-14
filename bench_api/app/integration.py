import sys
from typing import Optional

from app.config import MULTIKERNELBENCH_PATH, KERNELBENCH_PATH, REFERENCE_DIR


def register_kernelbench_dataset():
    """
    Scans KernelBench directory and adds functions to the shared dataset.
    """
    if str(MULTIKERNELBENCH_PATH) not in sys.path:
        sys.path.insert(0, str(MULTIKERNELBENCH_PATH))

    try:
        from dataset import dataset
    except ImportError:
        print("[WARNING] Could not import dataset from MultiKernelBench")
        return

    # Scan KernelBench
    if not KERNELBENCH_PATH.exists():
        print(f"[WARNING] KernelBench path does not exist: {KERNELBENCH_PATH}")
        return

    count = 0
    # KernelBench has level1, level2, level3, level4
    for level in ["level1", "level2", "level3", "level4"]:
        level_path = KERNELBENCH_PATH / level
        if not level_path.exists():
            continue

        for file in level_path.glob("*.py"):
            func_name = file.stem
            # Register in dataset if not already present
            if func_name not in dataset:
                dataset[func_name] = {"category": level, "source": "KernelBench"}
                count += 1

    print(f"[INFO] Registered {count} functions from KernelBench")


def get_dataset():
    """
    Returns the unified dataset containing both MultiKernelBench and KernelBench tasks.
    """
    if str(MULTIKERNELBENCH_PATH) not in sys.path:
        sys.path.insert(0, str(MULTIKERNELBENCH_PATH))

    try:
        from dataset import dataset

        # Ensure KernelBench is registered
        if not any(info.get("source") == "KernelBench" for info in dataset.values()):
            register_kernelbench_dataset()
        return dataset
    except ImportError:
        print("[WARNING] Could not import dataset from MultiKernelBench")
        return {}


def get_reference_path(function: str) -> Optional[str]:
    """
    Resolves the path to the reference implementation of a function.
    Checks dataset source to determine whether to look in MultiKernelBench or KernelBench.
    """
    if str(MULTIKERNELBENCH_PATH) not in sys.path:
        sys.path.insert(0, str(MULTIKERNELBENCH_PATH))

    try:
        from dataset import dataset
    except ImportError:
        return None

    if function not in dataset:
        return None

    info = dataset[function]
    category = info["category"]

    if info.get("source") == "KernelBench":
        path = KERNELBENCH_PATH / category / f"{function}.py"
    else:
        path = REFERENCE_DIR / category / f"{function}.py"

    return str(path) if path.exists() else None
