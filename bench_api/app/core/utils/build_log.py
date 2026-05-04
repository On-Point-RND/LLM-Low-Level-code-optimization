import re


def compact_build_log(message: str) -> str:
    """Extract errors from CUDA compilation logs."""
    lines = message.splitlines()
    if not lines:
        return message

    cu_errors = [l for l in lines if re.search(r"\.cu\(\d+\):\s*error:", l)]
    linker_lines = [l for l in lines if "undefined reference to" in l]

    if cu_errors:
        selected = cu_errors + linker_lines
    else:
        error_lines = [l for l in lines if re.search(r"error\s*:", l)]
        selected = error_lines + linker_lines

    if not selected:
        selected = [lines[0]]

    seen = set()
    result = []
    for l in selected:
        if l not in seen:
            seen.add(l)
            result.append(l)

    return "\n".join(result)
