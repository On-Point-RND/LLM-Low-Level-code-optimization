import argparse
import asyncio
import json
import os
import random
import re
import time
import tomllib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import openai
from openai import AsyncOpenAI


_AGENT_DIR = Path(__file__).parent


def _load_all_configs() -> dict:
    config_path = _AGENT_DIR / 'configs.toml'
    with open(config_path, 'rb') as f:
        return tomllib.load(f)


def load_config(name: str) -> dict:
    all_configs = _load_all_configs()
    if name not in all_configs:
        raise ValueError(f"Config '{name}' not found in configs.toml. Available: {list(all_configs)}")
    cfg = all_configs[name]
    if 'api_key' not in cfg:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                f"Config '{name}' has no 'api_key' and OPENAI_API_KEY env var is not set."
            )
        cfg['api_key'] = api_key
    return cfg


def load_search_config(method: str) -> dict:
    all_configs = _load_all_configs()
    if method not in all_configs:
        raise ValueError(f"Search method '{method}' not found in configs.toml. Available: {list(all_configs)}")
    return all_configs[method]


def load_stage_config() -> dict:
    """Return per-stage generation settings from [stage] section with defaults."""
    all_configs = _load_all_configs()
    defaults = {
        'diagnoser_temperature': 0.1,
        'advisor_temperature': 0.4,
        'diagnoser_max_tokens': 400,
        'advisor_max_tokens': 512,
        'coder_max_tokens': 16384,
    }
    return {**defaults, **all_configs.get('stage', {})}

experiment_logs = []

# Fail fast if bench_api is down; allow long kernel evaluation on GPU.
HTTPX_TIMEOUT = httpx.Timeout(connect=30.0, read=4800.0, write=600.0, pool=60.0)

# Second user turn if the model omits required tags (one retry per expansion).
FORMAT_RETRY_USER = (
    "Your previous reply did not contain valid non-empty <cuda_source>...</cuda_source> and "
    "<forward_body>...</forward_body> blocks, or <cuda_source> had no host binding "
    "`torch::Tensor name(torch::Tensor ...) { ... }`, or it contained PYBIND11_MODULE / "
    "nested `def forward`, or it was not a real CUDA kernel (need `__global__` and no "
    "torch::mm/matmul/conv ATen shortcuts). Reply again with ONLY a line starting with "
    "# Hypothesis: and exactly those two tagged blocks — no other text, no markdown code fences. "
    "Forward body is inserted into `forward(self, *args, **kwargs)` — use args[0], args[1], … "
    "for inputs from get_inputs()."
)

GEN_SEMAPHORE = asyncio.Semaphore(16)
EVAL_SEMAPHORE = asyncio.Semaphore(8)
KERNEL_SEMAPHORE = asyncio.Semaphore(1)
SAVE_LOCK = asyncio.Lock()

# Set to True at startup if vLLM supports n>1 sampling
VLLM_SUPPORTS_N_SAMPLING: bool = False


async def probe_n_sampling(client: AsyncOpenAI, model_name: str, n: int = 5) -> bool:
    """Check whether the inference server actually returns n completions when asked."""
    response = await _api_call_with_retry(lambda: client.chat.completions.create(
        model=model_name,
        messages=[{'role': 'user', 'content': 'Hi'}],
        max_tokens=1,
        n=n,
        temperature=1.0,
    ))
    got = len(response.choices)
    print(f'[probe] Requested n={n}, got {got} completions — {"supported" if got >= n else "NOT supported, will use parallel n=1 requests"}')
    return got >= n


async def get_hardware_info(client: httpx.AsyncClient | None = None) -> dict:
    url = os.getenv('BENCHAPI_URL', 'http://localhost:8123') + '/baseline'
    payload = {'language': 'cuda', 'function': 'relu'}
    if client:
        response = await client.post(url, json=payload)
    else:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as new_client:
            response = await new_client.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
    return {
        'hardware': data.get('hardware'),
        'compute_capability': data.get('compute_capability'),
    }


async def evaluate_kernel(
    code: str,
    language: str,
    function_name: str,
    client: httpx.AsyncClient | None = None,
    max_retries: int = 5,
    retry_delay: float = 10.0,
) -> dict:
    url = os.getenv('BENCHAPI_URL', 'http://localhost:8123') + '/evaluate'
    payload = {
        'language': language,
        'function': function_name,
        'function_code': code,
        'torch_compile_baseline': False,
        'include_baseline': False,
    }
    for attempt in range(max_retries):
        try:
            if client:
                response = await client.post(url, json=payload)
            else:
                async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as new_client:
                    response = await new_client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            if attempt == max_retries - 1:
                raise
            wait = retry_delay * (2 ** attempt)
            print(
                f'[{function_name}] bench_api network error (attempt {attempt + 1}/{max_retries}), '
                f'retrying in {wait:.0f}s: {type(e).__name__}: {e}',
                flush=True,
            )
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError as e:
            # Retry on 5xx (server errors), propagate 4xx immediately
            if e.response.status_code < 500 or attempt == max_retries - 1:
                raise
            wait = retry_delay * (2 ** attempt)
            print(
                f'[{function_name}] bench_api HTTP {e.response.status_code} '
                f'(attempt {attempt + 1}/{max_retries}), retrying in {wait:.0f}s',
                flush=True,
            )
            await asyncio.sleep(wait)


def _jinja_env() -> Environment:
    return Environment(loader=FileSystemLoader(_AGENT_DIR / 'prompts'))


def _last_result_needs_fix(last_result: dict | None) -> bool:
    """True when the previous bench step failed compile, correctness, or format — advisor should focus on fixes."""
    if last_result is None:
        return False
    return not last_result.get('compiled', False) or not last_result.get('valid', False)


def advisor_prompt(
    reference_code: str,
    hardware_info: str,
    compute_capability: str | None = None,
    last_code: str | None = None,
    last_result: dict | None = None,
    best_code: str | None = None,
    best_result: dict | None = None,
    history: list[dict] | None = None,
) -> str:
    """Optimization advisor — called only when the previous attempt was correct."""
    template = _jinja_env().get_template('kernel_advisor.md.jinja')
    return template.render(
        reference_code=reference_code,
        hardware_info=hardware_info,
        compute_capability=compute_capability,
        last_code=last_code,
        last_result=last_result or {},
        best_code=best_code,
        best_result=best_result or {},
        history=history or [],
    )


def diagnoser_prompt(
    reference_code: str,
    last_code: str,
    last_result: dict,
) -> str:
    template = _jinja_env().get_template('kernel_diagnoser.md.jinja')
    return template.render(
        reference_code=reference_code,
        last_code=last_code,
        last_result=last_result,
    )


def coder_prompt(
    reference_code: str,
    hardware_info: str,
    compute_capability: str | None = None,
    example_pre: str | None = None,
    example_after: str | None = None,
    last_code: str | None = None,
    last_result: dict | None = None,
    best_code: str | None = None,
    best_result: dict | None = None,
    history: list[dict] | None = None,
    advisor_text: str = '',
    diagnosis_text: str | None = None,
) -> str:
    template = _jinja_env().get_template('kernel_coder.md.jinja')
    return template.render(
        reference_code=reference_code,
        hardware_info=hardware_info,
        compute_capability=compute_capability,
        example_pre=example_pre,
        example_after=example_after,
        last_code=last_code,
        last_result=last_result or {},
        best_code=best_code,
        best_result=best_result or {},
        history=history or [],
        advisor_text=advisor_text,
        diagnosis_text=diagnosis_text,
    )


def _eval_result_to_feedback_text(result: dict | None) -> str:
    """Environment/bench outcome for prompts: compile / correctness / perf / profiler."""
    if not result:
        return 'No evaluation result.'
    lines: list[str] = []
    if not result.get('compiled', False):
        err = result.get('compilation_error') or ''
        lines.append(f'Compilation failed:\n{err[:8000]}')
        return '\n'.join(lines)
    if not result.get('valid', False):
        ci = result.get('correctness_info') or ''
        lines.append(f'Incorrect output:\n{ci[:8000]}')
        return '\n'.join(lines)
    perf = result.get('performance') or {}
    mean = perf.get('mean')
    if mean is not None:
        lines.append(f'Correct. Mean kernel time: {mean:.4g} ms')
    else:
        lines.append('Correct (no mean timing).')
    pi = result.get('performance_info')
    if pi:
        lines.append(f'Perf notes: {str(pi)[:2000]}')
    prof = result.get('cuda_profile')
    if prof:
        lines.append(f'Profiler mode: {result.get("profiling_mode") or "unknown"}')
        for k in prof[:12]:
            name = k.get('name', '?')
            us = k.get('self_cuda_us')
            cnt = k.get('count')
            fl = k.get('flops')
            lines.append(
                f'  - `{name}`: {float(us) / 1000.0:.3f} ms self_cuda, {cnt} calls'
                + (f', ~{fl} FLOPs' if fl else '')
            )
    return '\n'.join(lines)


def sample_one_sentence_advice_prompt(
    reference_code: str,
    generated_code: str,
    feedback_text: str,
) -> str:
    template = _jinja_env().get_template('kernel_sample_advice.md.jinja')
    return template.render(
        reference_code=reference_code,
        generated_code=generated_code,
        feedback_text=feedback_text,
    )


def sixth_aggregator_coder_prompt(
    reference_code: str,
    hardware_info: str,
    compute_capability: str | None,
    example_pre: str | None,
    example_after: str | None,
    samples: list[dict],
) -> str:
    """samples: {index, code, feedback_text (env), advice_sentence} — one triple per slot."""
    template = _jinja_env().get_template('kernel_sixth_coder.md.jinja')
    return template.render(
        reference_code=reference_code,
        hardware_info=hardware_info,
        compute_capability=compute_capability,
        example_pre=example_pre,
        example_after=example_after,
        samples=samples,
    )


def _prompt_messages_with_coder_reply(base_messages: list[dict], coder_raw: str) -> list[dict]:
    """Full chat transcript for parquet: base ends with coder user turn; append coder assistant."""
    return [*base_messages, {'role': 'assistant', 'content': coder_raw or ''}]


def _merge_llm_usage(u_first, u_second):
    """Sum token counts from advisor + coder completions."""
    if u_first is None:
        return u_second
    if u_second is None:
        return u_first

    def as_dict(u):
        if isinstance(u, dict):
            return u
        return u.model_dump() if hasattr(u, 'model_dump') else {}

    a = as_dict(u_first)
    b = as_dict(u_second)

    def pt(u):
        v = u.get('prompt_tokens')
        if v is None:
            v = u.get('input_tokens')
        return int(v) if v is not None else 0

    def ct(u):
        v = u.get('completion_tokens')
        if v is None:
            v = u.get('output_tokens')
        return int(v) if v is not None else 0

    merged = {**b}
    merged['prompt_tokens'] = pt(a) + pt(b)
    merged['completion_tokens'] = ct(a) + ct(b)
    return merged


def save_logs(exp_id: str, run_idx: int = 0):
    if not experiment_logs:
        return
    df = pd.DataFrame(experiment_logs)
    dir_path = Path(f'data/{exp_id}')
    dir_path.mkdir(parents=True, exist_ok=True)
    filename = dir_path / 'results.parquet'
    tmp = filename.with_suffix('.parquet.tmp')
    try:
        df.to_parquet(tmp, index=False)
        tmp.rename(filename)
        print(f'Logs saved to {filename}')
    except Exception as e:
        print(f'Failed to save logs to {filename}: {e}')


def _result_perf_mean_ms(result: dict | None) -> float | None:
    """Kernel timing mean from bench_api `performance` dict (milliseconds)."""
    if not result:
        return None
    perf = result.get('performance') or {}
    m = perf.get('mean')
    return float(m) if m is not None else None


def parse_hypothesis(code: str) -> str | None:
    """
    Extract the hypothesis comment from the top of generated code.

    Handles: wrong/missing 'Hypothesis:' label, multiple comment lines,
    blank lines before or between comment lines.
    Returns the hypothesis text, or None if no leading comments found.
    """
    lines = code.splitlines()

    # Skip leading blank lines
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1

    # Collect comment lines, tolerating blank lines between them
    comment_texts: list[str] = []
    j = i
    while j < len(lines):
        stripped = lines[j].strip()
        if stripped.startswith('#'):
            comment_texts.append(stripped[1:].strip())
            j += 1
        elif not stripped:
            # Blank line — include only if followed by another comment line
            k = j + 1
            while k < len(lines) and not lines[k].strip():
                k += 1
            if k < len(lines) and lines[k].strip().startswith('#'):
                j = k  # skip to next comment line
            else:
                break
        else:
            break  # hit real code

    if not comment_texts:
        return None

    full_text = ' '.join(comment_texts)

    # Strip optional "Hypothesis:" / "hypothesis:" / "HYPOTHESIS:" prefix
    full_text = re.sub(r'^hypothesis\s*:\s*', '', full_text, flags=re.IGNORECASE)

    return full_text.strip() or None


# ---------------------------------------------------------------------------
# Deterministic load_inline wrapper (Issue 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TorchBindings:
    """Forward declarations and function names for torch.utils.cpp_extension.load_inline."""

    declarations: list[str]
    function_names: list[str]


def _extract_torch_binding_signatures(cuda_source: str) -> TorchBindings:
    """Parse host-side bindings `torch::Tensor name(...)` from CUDA C++ source."""
    sigs: list[str] = []
    names: list[str] = []
    seen: set[str] = set()
    pos = 0
    pattern = re.compile(r"\b(?:static\s+)?torch::Tensor\s+(\w+)\s*\(")
    while True:
        m = pattern.search(cuda_source, pos)
        if not m:
            break
        abs_start = m.start()
        open_paren = m.end() - 1
        depth = 0
        j = open_paren
        while j < len(cuda_source):
            c = cuda_source[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    sig = cuda_source[abs_start : j + 1].strip()
                    name_m = re.match(r"^(?:static\s+)?torch::Tensor\s+(\w+)\s*\(", sig)
                    if name_m:
                        n = name_m.group(1)
                        if n not in seen:
                            seen.add(n)
                            sigs.append(sig)
                            names.append(n)
                    pos = j + 1
                    break
            j += 1
        else:
            break
    return TorchBindings(declarations=sigs, function_names=names)


# ATen / high-level ops in <cuda_source> — thin wrappers around cuBLAS/cuDNN, not custom CUDA.
_FORBIDDEN_ATEN_CUDA_RE = (
    re.compile(r"\btorch::mm\s*\("),
    re.compile(r"\btorch::bmm\s*\("),
    re.compile(r"\btorch::matmul\s*\("),
    re.compile(r"\btorch::addmm\s*\("),
    re.compile(r"\btorch::addmv\s*\("),
    re.compile(r"\btorch::baddbmm\s*\("),
    re.compile(r"\btorch::conv1d\s*\("),
    re.compile(r"\btorch::conv2d\s*\("),
    re.compile(r"\btorch::conv3d\s*\("),
    re.compile(r"\btorch::conv_transpose1d\s*\("),
    re.compile(r"\btorch::conv_transpose2d\s*\("),
    re.compile(r"\btorch::conv_transpose3d\s*\("),
    re.compile(r"\btorch::nn::functional::"),
    re.compile(r"\btorch::einsum\s*\("),
    re.compile(r"\bat::mm\s*\("),
    re.compile(r"\bat::matmul\s*\("),
    re.compile(r"\btorch::linalg::"),
)


def _validate_custom_cuda_policy(cuda_source: str) -> str | None:
    """
    Require at least one __global__ kernel and forbid obvious ATen one-call shortcuts.
    """
    if "__global__" not in cuda_source:
        return "cuda_source_missing_global_kernel"
    for rx in _FORBIDDEN_ATEN_CUDA_RE:
        if rx.search(cuda_source):
            return "cuda_source_forbidden_aten_shortcut"
    return None


_KERNEL_TEMPLATE = '''\
import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline as _load_inline

_ext = _load_inline(
    name={ext_name_repr},
    cpp_sources={cpp_sources_repr},
    cuda_sources={cuda_source_repr},
    functions={functions_repr},
    verbose=False,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ext = _ext

    def forward(self, *args, **kwargs):
{forward_body_indented}
'''


def assemble_kernel_module_or_error(response: str) -> tuple[str | None, str | None]:
    """
    Build the Python module from <cuda_source> and <forward_body> only (full text of LLM reply).

    Returns (module_code, None) on success, or (None, error_code) if tags are missing or invalid.
    Tags are matched case-insensitively on the full response (including inside markdown fences).
    """
    import hashlib

    if not response or not str(response).strip():
        return None, "empty_llm_response"

    cuda_match = re.search(r"(?i)<cuda_source>(.*?)</cuda_source>", response, re.DOTALL)
    fwd_match = re.search(r"(?i)<forward_body>(.*?)</forward_body>", response, re.DOTALL)

    if not cuda_match:
        return None, "missing_cuda_source_tags"
    if not fwd_match:
        return None, "missing_forward_body_tags"

    cuda_source = cuda_match.group(1).strip()
    forward_body = fwd_match.group(1).rstrip()

    if not cuda_source:
        return None, "empty_cuda_source"

    if "PYBIND11_MODULE" in cuda_source or "TORCH_EXTENSION_NAME" in cuda_source:
        return None, "cuda_source_pybind_forbidden"

    policy_err = _validate_custom_cuda_policy(cuda_source)
    if policy_err:
        return None, policy_err

    if re.search(r"\bdef\s+forward\s*\(", forward_body):
        return None, "forward_body_nested_def_forward"

    bindings = _extract_torch_binding_signatures(cuda_source)
    if not bindings.declarations or not bindings.function_names:
        return None, "no_torch_tensor_binding_in_cuda_source"

    cpp_decl = ";\n".join(bindings.declarations) + ";\n"

    if not forward_body.strip():
        forward_body = "pass"

    hyp_match = re.match(r"^\s*(#\s*Hypothesis:.*)", response, re.MULTILINE)
    hypothesis_line = (hyp_match.group(1).rstrip() + "\n") if hyp_match else ""

    ext_name = f"k_{hashlib.md5(cuda_source.encode()).hexdigest()[:8]}"
    indented = "\n".join(
        ("        " + line) if line.strip() else "" for line in forward_body.splitlines()
    )
    if not indented.strip():
        indented = "        pass"

    module_code = _KERNEL_TEMPLATE.format(
        ext_name_repr=repr(ext_name),
        cpp_sources_repr=repr(cpp_decl),
        cuda_source_repr=repr(cuda_source),
        functions_repr=repr(bindings.function_names),
        forward_body_indented=indented,
    )
    return hypothesis_line + module_code, None


def _preflight_assembled_module(code: str) -> str | None:
    """Return an error reason string if the assembled module cannot satisfy bench_api checks."""
    if "torch.utils.cpp_extension" not in code:
        return "preflight_missing_cpp_extension"
    if "_load_inline" not in code:
        return "preflight_missing_load_inline"
    if "class ModelNew" not in code:
        return "preflight_missing_ModelNew"
    return None


# Some models (e.g. MiniMax-M2.5) emit an internal reasoning block as
# `<think>...</think>` in message.content instead of a separate `reasoning`
# field (as gpt-oss does). Strip it so downstream parsers see only the answer.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def _strip_think_tags(content: str | None) -> str | None:
    """Remove <think>...</think> reasoning blocks from model output."""
    if not isinstance(content, str) or "<think" not in content.lower():
        return content
    cleaned = _THINK_BLOCK_RE.sub("", content)
    # Handle truncated reasoning (opening tag with no closing, or orphan closing tag).
    if _THINK_OPEN_RE.search(cleaned) and not _THINK_CLOSE_RE.search(cleaned):
        cleaned = _THINK_OPEN_RE.split(cleaned, maxsplit=1)[0]
    cleaned = _THINK_CLOSE_RE.split(cleaned, maxsplit=1)[-1]
    return cleaned.strip()


async def _api_call_with_retry(coro_fn, max_retries: int = 10, retry_delay: float = 5.0):
    """Retry an OpenAI-compatible API call on transient errors with exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = await coro_fn()
            if response is None or response.choices is None:
                raise openai.APIConnectionError(request=None)
            return response
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait = retry_delay * (2 ** attempt)  # exponential: 5, 10, 20, 40 …
            print(f'[retry] Rate limited (attempt {attempt + 1}/{max_retries}), retrying in {wait:.0f}s: {e}', flush=True)
            await asyncio.sleep(wait)
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            if attempt == max_retries - 1:
                raise
            wait = retry_delay * (2 ** min(attempt, 3))  # cap at retry_delay * 8
            print(f'[retry] Connection/timeout error (attempt {attempt + 1}/{max_retries}), retrying in {wait:.0f}s: {e}', flush=True)
            await asyncio.sleep(wait)
        except openai.InternalServerError as e:
            if attempt == max_retries - 1:
                raise
            wait = retry_delay * (2 ** min(attempt, 3))
            print(f'[retry] Server 5xx error (attempt {attempt + 1}/{max_retries}), retrying in {wait:.0f}s: {e}', flush=True)
            await asyncio.sleep(wait)


async def _generate_code(
    client: AsyncOpenAI,
    messages: list[dict],
    model_name: str,
    seed: int,
    temperature: float = 0.7,
    top_p: float = 0.8,
    n: int = 1,
    extra_body: dict | None = None,
    function_name: str = "kernel",
    max_tokens: int = 16384,
) -> tuple[list[str], any]:
    """Returns raw LLM message strings only; assembly is done in tree_search."""
    if n > 1 and not VLLM_SUPPORTS_N_SAMPLING:
        # Fall back to parallel n=1 requests; vLLM batches them via continuous batching
        async def _single(i):
            async with GEN_SEMAPHORE:
                return await _api_call_with_retry(lambda: client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    seed=(seed + i) % (2**32),
                    max_tokens=max_tokens,
                    n=1,
                    extra_body=extra_body,
                ))
        responses = await asyncio.gather(*[_single(i) for i in range(n)])
        raw_responses = [_strip_think_tags(r.choices[0].message.content) for r in responses]
        return raw_responses, responses[0].usage

    acquired = 0
    try:
        for _ in range(n):
            await GEN_SEMAPHORE.acquire()
            acquired += 1
        response = await _api_call_with_retry(lambda: client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_tokens=max_tokens,
            n=n,
            extra_body=extra_body,
        ))
    finally:
        for _ in range(acquired):
            GEN_SEMAPHORE.release()

    raw_responses = [_strip_think_tags(choice.message.content) for choice in response.choices]
    return raw_responses, response.usage


async def _evaluate_and_log(
    generated_code: str,
    prompt_messages: list[dict],
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int | None,
    t_gen_start: float,
    t_gen_end: float,
    model_name: str,
    base_seed: int,
    hardware_info: str,
    compute_capability: str | None,
    run_idx: int,
    run_seed: int,
    iteration_idx: int,
    iteration_seed: int,
    search_method: str,
    kernel_category: str,
    function_name: str,
    language: str,
    reference_code: str,
    dataset: str,
    node_id: int,
    parent_node_id: int | None,
    expansion_id: int,
    hypothesis: str | None,
    temperature: float = 0.7,
    top_p: float = 0.8,
    httpx_client: httpx.AsyncClient | None = None,
    skip_bench_eval: bool = False,
    format_error: str | None = None,
    advisor_text: str | None = None,
    advisor_prompt_text: str | None = None,
    diagnoser_text: str | None = None,
    diagnoser_prompt_text: str | None = None,
    generation_phase: str | None = None,
) -> dict:
    if skip_bench_eval:
        t_eval_start = t_gen_end
        t_eval_end = t_gen_end
        eval_result = {}
        compiled = False
        correctness = False
        compilation_message = format_error or "format_error"
        correctness_info = None
        performance_info = None
        timing = {}
        perf = {}
        profiling_mode = None
        cuda_profile = None
        print(
            f'[{function_name}] Node {node_id}: format_error (bench_api skipped): {compilation_message} | '
            f'{hypothesis or "(no hypothesis)"}',
            flush=True,
        )
    else:
        async with EVAL_SEMAPHORE:
            print(
                f'[{function_name}] bench_api /evaluate starting (node {node_id})...',
                flush=True,
            )
            t_eval_start = time.time()
            try:
                eval_result = await evaluate_kernel(generated_code, language, function_name, client=httpx_client)
            except Exception as e:
                t_eval_end = time.time()
                print(
                    f'[{function_name}] bench_api /evaluate FAILED after retries '
                    f'(node {node_id}): {type(e).__name__}: {e}',
                    flush=True,
                )
                # Treat as a non-compiled result so the tree can continue with other nodes
                eval_result = {
                    'compiled': False,
                    'correctness': None,
                    'compile_info': f'bench_api_error: {type(e).__name__}: {e}',
                }
            else:
                t_eval_end = time.time()
                print(
                    f'[{function_name}] bench_api /evaluate finished in {t_eval_end - t_eval_start:.1f}s',
                    flush=True,
                )

        compiled = bool(eval_result.get('compiled') or False)
        correctness = eval_result.get('correctness')
        compilation_message = eval_result.get('compile_info')
        correctness_info = eval_result.get('correctness_info')
        performance_info = eval_result.get('performance_info')
        timing = eval_result.get('timing') or {}
        perf = eval_result.get('performance') or {}
        profiling_mode = eval_result.get('profiling_mode')
        cuda_profile = eval_result.get('cuda_profile')

        mean_ms = perf.get('mean') if isinstance(perf, dict) else None
        mean_s = f"{float(mean_ms):.3g}ms" if mean_ms is not None else "N/A"
        print(
            f'[{function_name}] Node {node_id}: Compiled: {compiled}, Valid: {correctness}, '
            f'Mean time: {mean_s} | {hypothesis or "(no hypothesis)"}'
        )

    experiment_logs.append(
        {
            'model_name': model_name,
            'temperature': temperature,
            'top_p': top_p,
            'global_seed': base_seed,
            'hardware_info': hardware_info,
            'compute_capability': compute_capability,
            'search_method': search_method,
            'run_id': run_idx,
            'run_seed': run_seed,
            'iteration': iteration_idx,
            'iteration_seed': iteration_seed,
            'timestamp_start': datetime.fromtimestamp(t_gen_start),
            'generation_timing': t_gen_end - t_gen_start,
            'evaluation_timing_compilation': timing.get('compilation'),
            'evaluation_timing_correctness': timing.get('correctness'),
            'evaluation_timing_performance': timing.get('performance'),
            'evaluation_timing_total': timing.get('total'),
            'duration_total': (t_gen_end - t_gen_start) if skip_bench_eval else (t_eval_end - t_gen_start),
            'kernel_category': kernel_category,
            'kernel_name': function_name,
            'language': language,
            'reference_code': reference_code,
            'dataset': dataset,
            'generated_code': generated_code,
            'prompts': json.dumps(prompt_messages, ensure_ascii=False),
            'compiled': compiled,
            'correctness': correctness,
            'correctness_info': correctness_info,
            'performance_mean': perf.get('mean'),
            'performance_std': perf.get('std'),
            'performance_min': perf.get('min'),
            'performance_max': perf.get('max'),
            'performance_num_trials': perf.get('num_trials'),
            'performance_info': performance_info,
            'profiling_mode': profiling_mode,
            'cuda_profile': json.dumps(cuda_profile) if cuda_profile else None,
            'compilation_message': compilation_message,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cached_tokens': cached_tokens,
            'node_id': node_id,
            'parent_node_id': parent_node_id,
            'expansion_id': expansion_id,
            'hypothesis': hypothesis,
            'advisor_prompt': advisor_prompt_text,
            'advisor_text': advisor_text,
            'diagnoser_prompt': diagnoser_prompt_text,
            'diagnoser_text': diagnoser_text,
            'generation_phase': generation_phase,
        }
    )

    return {
        'code': generated_code,
        'compiled': compiled,
        'compilation_error': compilation_message,
        'valid': correctness,
        'correctness_info': correctness_info,
        'performance_info': performance_info,
        'performance': perf,          # dict {mean, std, min, max, num_trials} in ms
        'profiling_mode': profiling_mode,  # "full" | "fast" | None
        'cuda_profile': cuda_profile,      # list[{name, self_cuda_us, count, flops}] | None
    }


@dataclass
class TreeNode:
    messages: list[dict]
    depth: int
    parent: 'TreeNode | None' = None
    node_id: int | None = None  # assigned when the node is logged
    code: str = ''
    raw_response: str = ''
    hypothesis: str | None = None
    advisor_text: str | None = None
    result: dict | None = None


class NodeSelector(ABC):
    @abstractmethod
    def select(self, candidates: list[TreeNode]) -> list[TreeNode]:
        """Select which nodes to expand at the next tree level."""
        ...


class GreedySelector(NodeSelector):
    """Keep the top-k nodes ranked by evaluation score."""

    def __init__(self, beam_size: int = 1):
        self.beam_size = beam_size

    def select(self, candidates: list[TreeNode]) -> list[TreeNode]:
        def score(node: TreeNode) -> tuple:
            r = node.result or {}
            if r.get('valid'):
                m = _result_perf_mean_ms(r)
                # Lower mean latency is better; sort key higher = better → use -mean (missing → slow).
                return (2, -(m if m is not None else 1e9))
            elif r.get('compiled'):
                return (1, 0.0)
            return (0, 0.0)

        return sorted(candidates, key=score, reverse=True)[: self.beam_size]


def _collect_ancestry(node: TreeNode) -> tuple[str | None, dict | None, str | None, dict | None, list[dict]]:
    """
    Build prompt context from a node's ancestry chain.

    Returns (last_code, last_result, best_code, best_result, history) where:
    - last_code / last_result: node's own code and evaluation result (most recent attempt)
    - best_code / best_result: best attempt so far (lowest measured mean time among correct, if any)
    - history: compact list of {hypothesis, result} for all older ancestors,
      oldest first (excluding last node and best node to avoid duplication)
    """
    if node.parent is None:
        # node is root — no prior attempts
        return None, None, None, None, []

    # Collect path from root's child down to node (inclusive)
    path: list[TreeNode] = []
    current = node
    while current.parent is not None:
        path.append(current)
        current = current.parent
    path.reverse()  # oldest first

    last = path[-1]

    def score_node(n: TreeNode) -> tuple:
        r = n.result or {}
        if not r.get('compiled') or not r.get('valid'):
            return (0, 0.0)
        m = _result_perf_mean_ms(r)
        return (1, -(m if m is not None else 1e9))

    best = max(path, key=score_node)

    # History: all ancestors except last and best (avoid duplication)
    history_nodes = [n for n in path[:-1] if n != best]
    history = [
        {'hypothesis': n.hypothesis, 'advisor_text': n.advisor_text, 'result': n.result}
        for n in history_nodes
    ]

    return last.code, last.result, best.code, best.result, history


async def tree_search(
    client: AsyncOpenAI,
    model_name: str,
    reference_code: str,
    hardware_info: str,
    compute_capability: str | None,
    example_pre: str,
    example_after: str,
    language: str,
    function_name: str,
    run_idx: int,
    run_seed: int,
    width: int,
    height: int,
    selector: NodeSelector,
    kernel_category: str,
    dataset: str,
    search_method_name: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    httpx_client: httpx.AsyncClient | None = None,
    two_stage: bool = True,
    diagnoser_temperature: float = 0.1,
    advisor_temperature: float = 0.4,
    diagnoser_max_tokens: int = 400,
    advisor_max_tokens: int = 512,
    coder_max_tokens: int = 16384,
    log_generation_phase: str | None = None,
):
    """
    Generic tree search over kernel generations.

    width=1, height=N  -> iterative refinement (single path, sequential feedback)
    width=N, height=1  -> multiple sampling (N parallel candidates, no feedback)
    width=W, height=H  -> beam / tree search (W children per node, H levels deep)

    At each level the selector chooses which evaluated nodes to expand next.

    two_stage: if True, diagnoser (on failure) or advisor (on success) runs before the coder.
    """
    print(f'\n=== {search_method_name.replace("_", " ").title()}: Run {run_idx + 1} (Seed: {run_seed}) ===')

    root = TreeNode(messages=[], depth=0)
    frontier = [root]
    eval_counter = 0
    expansion_counter = 0
    last_layer_leaves: list[TreeNode] = []

    for depth in range(height):
        print(f'--- Depth {depth + 1}/{height}, frontier size: {len(frontier)}, branching x{width} ---')

        async def process_parent(parent, node_expansion_id, node_eval_id_start):
            last_code, last_result, best_code, best_result, history = _collect_ancestry(parent)

            # Vary seed per expansion call so repeated expansion of the same node
            # (exploitation) produces diverse children rather than identical copies.
            call_seed = (run_seed + node_expansion_id) % (2**32)

            system_msg = {'role': 'system', 'content': 'You are a helpful assistant'}
            usage_advisor = None
            usage_diagnoser = None
            advisor_text_out: str | None = None
            diagnosis_text_out: str | None = None
            advisor_user: str = ''
            diagnoser_user: str = ''

            t_gen_start = time.time()

            needs_fix = _last_result_needs_fix(last_result)

            if two_stage and last_code and needs_fix:
                # Stage 0 — DIAGNOSER: previous attempt failed.
                # Read the broken code + error and output a specific bug report.
                # Advisor is NOT called in this branch.
                diagnoser_user = diagnoser_prompt(
                    reference_code=reference_code,
                    last_code=last_code,
                    last_result=last_result or {},
                )
                print(
                    f'[{function_name}] LLM diagnoser (stage 0, fix path) started (depth {depth + 1}/{height})...',
                    flush=True,
                )
                raw_diagnoser, usage_diagnoser = await _generate_code(
                    client=client,
                    messages=[system_msg, {'role': 'user', 'content': diagnoser_user}],
                    model_name=model_name,
                    seed=call_seed,
                    temperature=diagnoser_temperature,
                    top_p=top_p,
                    n=1,
                    extra_body={'cache_salt': f'{function_name}-{run_seed}-diagnoser'},
                    function_name=function_name,
                    max_tokens=diagnoser_max_tokens,
                )
                diagnosis_text_out = (raw_diagnoser[0] or '').strip() or None
                print(
                    f'[{function_name}] LLM diagnoser (stage 0) finished ({len(diagnosis_text_out or "")} chars)',
                    flush=True,
                )

            elif two_stage and not needs_fix:
                # Stage 1 — ADVISOR: previous attempt was correct (or first attempt).
                # Focus purely on speed optimizations. Diagnoser is NOT called in this branch.
                advisor_user = advisor_prompt(
                    reference_code=reference_code,
                    hardware_info=hardware_info,
                    compute_capability=compute_capability,
                    last_code=last_code,
                    last_result=last_result,
                    best_code=best_code,
                    best_result=best_result,
                    history=history,
                )
                print(
                    f'[{function_name}] LLM advisor (stage 1, optimize path) started (depth {depth + 1}/{height})...',
                    flush=True,
                )
                raw_advisor, usage_advisor = await _generate_code(
                    client=client,
                    messages=[system_msg, {'role': 'user', 'content': advisor_user}],
                    model_name=model_name,
                    seed=call_seed,
                    temperature=advisor_temperature,
                    top_p=top_p,
                    n=1,
                    extra_body={'cache_salt': f'{function_name}-{run_seed}-advisor'},
                    function_name=function_name,
                    max_tokens=advisor_max_tokens,
                )
                advisor_text_out = (raw_advisor[0] or '').strip()
                print(
                    f'[{function_name}] LLM advisor (stage 1) finished ({len(advisor_text_out)} chars)',
                    flush=True,
                )

            coder_user = coder_prompt(
                reference_code=reference_code,
                hardware_info=hardware_info,
                compute_capability=compute_capability,
                example_pre=example_pre,
                example_after=example_after,
                last_code=last_code,
                last_result=last_result,
                best_code=best_code,
                best_result=best_result,
                history=history,
                advisor_text=advisor_text_out or '',
                diagnosis_text=diagnosis_text_out,
            )

            if two_stage and advisor_text_out:
                # Optimization path: advisor turn is in context
                messages = [
                    system_msg,
                    {'role': 'user', 'content': advisor_user},
                    {'role': 'assistant', 'content': advisor_text_out},
                    {'role': 'user', 'content': coder_user},
                ]
            else:
                # Fix path (diagnoser) or no two_stage: coder sees its own prompt only
                messages = [
                    system_msg,
                    {'role': 'user', 'content': coder_user},
                ]

            base_coder_messages = [system_msg, {'role': 'user', 'content': coder_user}]
            attempt_messages = base_coder_messages
            raw_responses: list[str] = []
            usage_coder = None

            for format_attempt in range(2):
                gen_label = (
                    f'LLM coder (stage 2) started (depth {depth + 1}/{height})'
                    if two_stage
                    else f'LLM generation started (depth {depth + 1}/{height})'
                )
                print(f'[{function_name}] {gen_label}...', flush=True)
                raw_responses, usage_coder = await _generate_code(
                    client=client,
                    messages=attempt_messages,
                    model_name=model_name,
                    seed=call_seed,
                    temperature=temperature,
                    top_p=top_p,
                    n=width,
                    extra_body={'cache_salt': f'{function_name}-{run_seed}'},
                    function_name=function_name,
                    max_tokens=coder_max_tokens,
                )
                done_label = (
                    f'LLM coder (stage 2) finished'
                    if two_stage
                    else f'LLM generation finished'
                )
                print(
                    f'[{function_name}] {done_label} in {time.time() - t_gen_start:.1f}s',
                    flush=True,
                )

                assembled_list: list[str | None] = []
                err_list: list[str | None] = []
                all_ok = True
                for raw in raw_responses:
                    mod, err = assemble_kernel_module_or_error(raw)
                    if mod:
                        pe = _preflight_assembled_module(mod)
                        if pe:
                            mod, err = None, pe
                    if mod is None:
                        all_ok = False
                    assembled_list.append(mod)
                    err_list.append(err)

                if all_ok:
                    break
                if format_attempt == 0:
                    print(
                        f'[{function_name}] Missing/invalid <cuda_source>/<forward_body> or preflight; '
                        f'retrying once with format reminder...',
                        flush=True,
                    )
                    attempt_messages = base_coder_messages + [{'role': 'user', 'content': FORMAT_RETRY_USER}]
                else:
                    break

            t_gen_end = time.time()
            usage = _merge_llm_usage(_merge_llm_usage(usage_diagnoser, usage_advisor), usage_coder)

            u = usage if isinstance(usage, dict) else (usage.model_dump() if usage else {})
            pt_raw = u.get('prompt_tokens')
            ct_raw = u.get('completion_tokens')
            if pt_raw is None:
                pt_raw = u.get('input_tokens')
            if ct_raw is None:
                ct_raw = u.get('output_tokens')
            prompt_tokens = 0 if pt_raw is None else int(pt_raw)
            completion_tokens = 0 if ct_raw is None else int(ct_raw)
            cached_tokens = (u.get('prompt_tokens_details') or {}).get('cached_tokens')
            if pt_raw is None and ct_raw is None:
                print(
                    f'[{function_name}] Warning: LLM returned no token usage; logging 0. Raw usage={u!r}',
                    flush=True,
                )

            n_bench = sum(1 for c in assembled_list if c is not None)
            print(
                f'[{function_name}] Scheduling {n_bench} bench_api /evaluate call(s)...',
                flush=True,
            )

            eval_tasks = []
            for i, (code, raw, fmt_err) in enumerate(zip(assembled_list, raw_responses, err_list)):
                hyp_src = code if code else (raw or '')
                hypothesis = parse_hypothesis(hyp_src)
                if code is None:
                    eval_tasks.append(
                        _evaluate_and_log(
                            generated_code=raw or '',
                            prompt_messages=_prompt_messages_with_coder_reply(messages, raw or ''),
                            input_tokens=prompt_tokens if i == 0 else 0,
                            output_tokens=completion_tokens if i == 0 else 0,
                            cached_tokens=cached_tokens if i == 0 else 0,
                            t_gen_start=t_gen_start,
                            t_gen_end=t_gen_end,
                            model_name=model_name,
                            base_seed=run_seed,
                            hardware_info=hardware_info,
                            compute_capability=compute_capability,
                            run_idx=run_idx,
                            run_seed=run_seed,
                            iteration_idx=node_eval_id_start + i,
                            iteration_seed=call_seed,
                            search_method=search_method_name,
                            kernel_category=kernel_category,
                            function_name=function_name,
                            language=language,
                            reference_code=reference_code,
                            dataset=dataset,
                            node_id=node_eval_id_start + i,
                            parent_node_id=parent.node_id,
                            expansion_id=node_expansion_id,
                            hypothesis=hypothesis,
                            temperature=temperature,
                            top_p=top_p,
                            httpx_client=httpx_client,
                            skip_bench_eval=True,
                            format_error=fmt_err or 'format_error',
                            advisor_text=advisor_text_out,
                            advisor_prompt_text=advisor_user or None,
                            diagnoser_text=diagnosis_text_out,
                            diagnoser_prompt_text=diagnoser_user if (two_stage and last_code and needs_fix) else None,
                            generation_phase=log_generation_phase,
                        )
                    )
                else:
                    eval_tasks.append(
                        _evaluate_and_log(
                            generated_code=code,
                            prompt_messages=_prompt_messages_with_coder_reply(messages, raw or ''),
                            input_tokens=prompt_tokens if i == 0 else 0,
                            output_tokens=completion_tokens if i == 0 else 0,
                            cached_tokens=cached_tokens if i == 0 else 0,
                            t_gen_start=t_gen_start,
                            t_gen_end=t_gen_end,
                            model_name=model_name,
                            base_seed=run_seed,
                            hardware_info=hardware_info,
                            compute_capability=compute_capability,
                            run_idx=run_idx,
                            run_seed=run_seed,
                            iteration_idx=node_eval_id_start + i,
                            iteration_seed=call_seed,
                            search_method=search_method_name,
                            kernel_category=kernel_category,
                            function_name=function_name,
                            language=language,
                            reference_code=reference_code,
                            dataset=dataset,
                            node_id=node_eval_id_start + i,
                            parent_node_id=parent.node_id,
                            expansion_id=node_expansion_id,
                            hypothesis=hypothesis,
                            temperature=temperature,
                            top_p=top_p,
                            httpx_client=httpx_client,
                            advisor_text=advisor_text_out,
                            advisor_prompt_text=advisor_user or None,
                            diagnoser_text=diagnosis_text_out,
                            diagnoser_prompt_text=diagnoser_user if (two_stage and last_code and needs_fix) else None,
                            generation_phase=log_generation_phase,
                        )
                    )

            eval_results = await asyncio.gather(*eval_tasks, return_exceptions=True)

            children = []
            for i, (result, code, raw) in enumerate(zip(eval_results, assembled_list, raw_responses)):
                if isinstance(result, BaseException):
                    print(
                        f'[{function_name}] Node {node_eval_id_start + i} eval task failed: '
                        f'{type(result).__name__}: {result}',
                        flush=True,
                    )
                    result = {'compiled': False, 'correctness': False,
                              'compile_info': f'task_error: {type(result).__name__}: {result}'}
                child = TreeNode(
                    messages=messages,
                    depth=depth + 1,
                    parent=parent,
                    node_id=node_eval_id_start + i,
                    code=code or (raw or ''),
                    raw_response=raw,
                    hypothesis=parse_hypothesis(code or raw or ''),
                    advisor_text=advisor_text_out,
                    result=result,
                )
                children.append(child)
            return children

        expansion_tasks = []
        for parent in frontier:
            expansion_tasks.append(process_parent(parent, expansion_counter, eval_counter))
            expansion_counter += 1
            eval_counter += width

        expansion_results = await asyncio.gather(*expansion_tasks, return_exceptions=True)
        next_candidates = []
        for branch in expansion_results:
            if isinstance(branch, BaseException):
                print(f'[{function_name}] Expansion task failed: {type(branch).__name__}: {branch}', flush=True)
            else:
                next_candidates.extend(branch)

        last_layer_leaves = next_candidates

        if depth < height - 1:
            selected = selector.select(next_candidates)
            if not selected:
                print('Selector returned empty frontier, stopping early.')
                break
            frontier = selected

    return last_layer_leaves


def _first_sentence_line(text: str) -> str:
    line = (text or '').strip().split('\n', 1)[0].strip()
    return line[:4000]


def _append_sampling_advice_log(
    *,
    model_name: str,
    temperature: float,
    top_p: float,
    run_seed: int,
    run_idx: int,
    search_method: str,
    kernel_category: str,
    function_name: str,
    language: str,
    reference_code: str,
    dataset: str,
    hardware_info: str,
    compute_capability: str | None,
    advice_prompt: str,
    advice_text: str,
    node_id: int,
    iteration_seed: int,
    t_gen_start: float,
    t_gen_end: float,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int | None,
) -> None:
    system_msg = {'role': 'system', 'content': 'You are a helpful assistant'}
    experiment_logs.append(
        {
            'model_name': model_name,
            'temperature': temperature,
            'top_p': top_p,
            'global_seed': run_seed,
            'hardware_info': hardware_info,
            'compute_capability': compute_capability,
            'search_method': search_method,
            'run_id': run_idx,
            'run_seed': run_seed,
            'iteration': node_id,
            'iteration_seed': iteration_seed,
            'timestamp_start': datetime.fromtimestamp(t_gen_start),
            'generation_timing': t_gen_end - t_gen_start,
            'evaluation_timing_compilation': None,
            'evaluation_timing_correctness': None,
            'evaluation_timing_performance': None,
            'evaluation_timing_total': None,
            'duration_total': t_gen_end - t_gen_start,
            'kernel_category': kernel_category,
            'kernel_name': function_name,
            'language': language,
            'reference_code': reference_code,
            'dataset': dataset,
            'generated_code': advice_text,
            'prompts': json.dumps(
                [system_msg, {'role': 'user', 'content': advice_prompt}],
                ensure_ascii=False,
            ),
            'compiled': False,
            'correctness': None,
            'correctness_info': None,
            'performance_mean': None,
            'performance_std': None,
            'performance_min': None,
            'performance_max': None,
            'performance_num_trials': None,
            'performance_info': None,
            'profiling_mode': None,
            'cuda_profile': None,
            'compilation_message': None,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cached_tokens': cached_tokens,
            'node_id': node_id,
            'parent_node_id': None,
            'expansion_id': node_id,
            'hypothesis': None,
            'advisor_prompt': None,
            'advisor_text': None,
            'diagnoser_prompt': None,
            'diagnoser_text': None,
            'generation_phase': 'sampling_advice',
        }
    )


async def sampling_feedback_sixth_search(
    client: AsyncOpenAI,
    model_name: str,
    reference_code: str,
    hardware_info: str,
    compute_capability: str | None,
    example_pre: str,
    example_after: str,
    language: str,
    function_name: str,
    run_idx: int,
    run_seed: int,
    width: int,
    height: int,
    selector: NodeSelector,
    kernel_category: str,
    dataset: str,
    search_method_name: str,
    temperature: float,
    top_p: float,
    httpx_client: httpx.AsyncClient | None,
    diagnoser_temperature: float,
    advisor_temperature: float,
    diagnoser_max_tokens: int,
    advisor_max_tokens: int,
    coder_max_tokens: int,
    sixth_max_attempts: int,
    advice_temperature: float,
    advice_max_tokens: int,
) -> None:
    """
    Phase A: `width` independent coders (no shared advisor/diagnoser) → bench → environment feedback each.
    Phase B: `width` independent one-sentence LLM advices (code + env + reference per slot).
    Phase C: one prompt = 5 codes + 5 env texts + 5 advices + reference → candidate #6, up to `sixth_max_attempts` tries.
    """
    system_msg = {'role': 'system', 'content': 'You are a helpful assistant'}

    leaves = await tree_search(
        client=client,
        model_name=model_name,
        reference_code=reference_code,
        hardware_info=hardware_info,
        compute_capability=compute_capability,
        example_pre=example_pre,
        example_after=example_after,
        language=language,
        function_name=function_name,
        run_idx=run_idx,
        run_seed=run_seed,
        width=width,
        height=height,
        selector=selector,
        kernel_category=kernel_category,
        dataset=dataset,
        search_method_name=search_method_name,
        temperature=temperature,
        top_p=top_p,
        httpx_client=httpx_client,
        two_stage=False,
        diagnoser_temperature=diagnoser_temperature,
        advisor_temperature=advisor_temperature,
        diagnoser_max_tokens=diagnoser_max_tokens,
        advisor_max_tokens=advisor_max_tokens,
        coder_max_tokens=coder_max_tokens,
        log_generation_phase='sampling_batch5',
    )

    batch_leaves = sorted(
        [n for n in leaves if n.node_id is not None],
        key=lambda n: int(n.node_id),
    )
    if len(batch_leaves) < width:
        print(
            f'[{function_name}] sampling sixth: expected {width} batch nodes, got {len(batch_leaves)}, aborting sixth phase.',
            flush=True,
        )
        return

    async def one_advice(ix: int, node: TreeNode) -> tuple[str, object | None]:
        fb = _eval_result_to_feedback_text(node.result)
        code = node.code or node.raw_response or ''
        prompt = sample_one_sentence_advice_prompt(reference_code, code, fb)
        call_seed = (run_seed + 50 + ix) % (2**32)
        t_a0 = time.time()
        raw_list, usage = await _generate_code(
            client=client,
            messages=[system_msg, {'role': 'user', 'content': prompt}],
            model_name=model_name,
            seed=call_seed,
            temperature=advice_temperature,
            top_p=top_p,
            n=1,
            extra_body={'cache_salt': f'{function_name}-{run_seed}-sample-advice-{ix}'},
            function_name=function_name,
            max_tokens=advice_max_tokens,
        )
        t_a1 = time.time()
        advice = _first_sentence_line(raw_list[0] if raw_list else '')
        u = usage if isinstance(usage, dict) else (usage.model_dump() if usage else {})
        pt_raw = u.get('prompt_tokens')
        ct_raw = u.get('completion_tokens')
        if pt_raw is None:
            pt_raw = u.get('input_tokens')
        if ct_raw is None:
            ct_raw = u.get('output_tokens')
        pt = 0 if pt_raw is None else int(pt_raw)
        ct = 0 if ct_raw is None else int(ct_raw)
        cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens')
        _append_sampling_advice_log(
            model_name=model_name,
            temperature=advice_temperature,
            top_p=top_p,
            run_seed=run_seed,
            run_idx=run_idx,
            search_method=search_method_name,
            kernel_category=kernel_category,
            function_name=function_name,
            language=language,
            reference_code=reference_code,
            dataset=dataset,
            hardware_info=hardware_info,
            compute_capability=compute_capability,
            advice_prompt=prompt,
            advice_text=advice,
            node_id=50 + ix,
            iteration_seed=call_seed,
            t_gen_start=t_a0,
            t_gen_end=t_a1,
            input_tokens=pt,
            output_tokens=ct,
            cached_tokens=cached,
        )
        return advice, usage

    advice_results = await asyncio.gather(*[one_advice(i, n) for i, n in enumerate(batch_leaves)])
    advice_sentences = [a[0] for a in advice_results]

    samples_payload = []
    for i, node in enumerate(batch_leaves):
        samples_payload.append(
            {
                'index': i + 1,
                'code': node.code or node.raw_response or '',
                'feedback_text': _eval_result_to_feedback_text(node.result),
                'advice_sentence': advice_sentences[i],
            }
        )

    sixth_user = sixth_aggregator_coder_prompt(
        reference_code=reference_code,
        hardware_info=hardware_info,
        compute_capability=compute_capability,
        example_pre=example_pre,
        example_after=example_after,
        samples=samples_payload,
    )

    for attempt in range(sixth_max_attempts):
        call_seed = (run_seed + 200 + attempt * 31) % (2**32)
        base_coder_messages = [system_msg, {'role': 'user', 'content': sixth_user}]
        attempt_messages = base_coder_messages
        raw_response = ''
        usage_coder = None
        t_gen_start = time.time()
        mod = None
        err = None

        for format_attempt in range(2):
            print(
                f'[{function_name}] Sixth candidate attempt {attempt + 1}/{sixth_max_attempts} '
                f'(format round {format_attempt + 1})...',
                flush=True,
            )
            raw_list, usage_coder = await _generate_code(
                client=client,
                messages=attempt_messages,
                model_name=model_name,
                seed=call_seed,
                temperature=temperature,
                top_p=top_p,
                n=1,
                extra_body={'cache_salt': f'{function_name}-{run_seed}-sixth-{attempt}'},
                function_name=function_name,
                max_tokens=coder_max_tokens,
            )
            raw_response = raw_list[0] if raw_list else ''
            mod, err = assemble_kernel_module_or_error(raw_response)
            if mod:
                pe = _preflight_assembled_module(mod)
                if pe:
                    mod, err = None, pe
            if mod is not None:
                break
            if format_attempt == 0:
                print(
                    f'[{function_name}] Sixth attempt {attempt + 1}: invalid tags/preflight ({err}); '
                    f'retrying with format reminder...',
                    flush=True,
                )
                attempt_messages = base_coder_messages + [{'role': 'user', 'content': FORMAT_RETRY_USER}]
            else:
                break

        t_gen_end = time.time()
        u = usage_coder if isinstance(usage_coder, dict) else (
            usage_coder.model_dump() if usage_coder else {}
        )
        pt_raw = u.get('prompt_tokens')
        ct_raw = u.get('completion_tokens')
        if pt_raw is None:
            pt_raw = u.get('input_tokens')
        if ct_raw is None:
            ct_raw = u.get('output_tokens')
        prompt_tokens = 0 if pt_raw is None else int(pt_raw)
        completion_tokens = 0 if ct_raw is None else int(ct_raw)
        cached_tokens = (u.get('prompt_tokens_details') or {}).get('cached_tokens')

        node_id = 5 + attempt
        hyp_src = raw_response
        hypothesis = parse_hypothesis(hyp_src)

        if mod is None:
            await _evaluate_and_log(
                generated_code=raw_response,
                prompt_messages=_prompt_messages_with_coder_reply(base_coder_messages, raw_response),
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                t_gen_start=t_gen_start,
                t_gen_end=t_gen_end,
                model_name=model_name,
                base_seed=run_seed,
                hardware_info=hardware_info,
                compute_capability=compute_capability,
                run_idx=run_idx,
                run_seed=run_seed,
                iteration_idx=node_id,
                iteration_seed=call_seed,
                search_method=search_method_name,
                kernel_category=kernel_category,
                function_name=function_name,
                language=language,
                reference_code=reference_code,
                dataset=dataset,
                node_id=node_id,
                parent_node_id=None,
                expansion_id=600 + attempt,
                hypothesis=hypothesis,
                temperature=temperature,
                top_p=top_p,
                httpx_client=httpx_client,
                skip_bench_eval=True,
                format_error=err or 'format_error',
                generation_phase='sampling_sixth',
            )
            print(
                f'[{function_name}] Sixth attempt {attempt + 1}: still not assembleable; continuing.',
                flush=True,
            )
            continue

        eval_out = await _evaluate_and_log(
            generated_code=mod,
            prompt_messages=_prompt_messages_with_coder_reply(base_coder_messages, raw_response),
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            t_gen_start=t_gen_start,
            t_gen_end=t_gen_end,
            model_name=model_name,
            base_seed=run_seed,
            hardware_info=hardware_info,
            compute_capability=compute_capability,
            run_idx=run_idx,
            run_seed=run_seed,
            iteration_idx=node_id,
            iteration_seed=call_seed,
            search_method=search_method_name,
            kernel_category=kernel_category,
            function_name=function_name,
            language=language,
            reference_code=reference_code,
            dataset=dataset,
            node_id=node_id,
            parent_node_id=None,
            expansion_id=600 + attempt,
            hypothesis=hypothesis,
            temperature=temperature,
            top_p=top_p,
            httpx_client=httpx_client,
            generation_phase='sampling_sixth',
        )
        if eval_out.get('compiled') and eval_out.get('valid'):
            print(
                f'[{function_name}] Sixth candidate succeeded on attempt {attempt + 1} '
                f'(compiled + correct).',
                flush=True,
            )
            return
        print(
            f'[{function_name}] Sixth attempt {attempt + 1}: compiled={eval_out.get("compiled")} '
            f'valid={eval_out.get("valid")}; retrying if attempts remain.',
            flush=True,
        )

    print(
        f'[{function_name}] Sixth candidate: exhausted {sixth_max_attempts} attempt(s) without valid+correct.',
        flush=True,
    )


GLOBAL_SEED = 2026
# Set by main() from the loaded config
MODEL_NAME = None
VLLM_BASE_URL = None


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp-id', type=str, required=True, help='Experiment ID for logging and resumption'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='kernelbench',
        help='Dataset to use (default: kernelbench)',
    )
    parser.add_argument(
        '--search-method',
        type=str,
        required=True,
        help='Search method config name from configs.toml (e.g. iterative, sampling)',
    )
    parser.add_argument(
        '--kernels',
        type=str,
        help='Comma-separated list of kernel names to run (e.g. add,relu)',
    )
    parser.add_argument(
        '--parallel-kernels',
        type=int,
        default=1,
        help='Number of kernels to search in parallel (default: 1)',
    )
    parser.add_argument(
        '--categories',
        type=str,
        help='Comma-separated list of kernel categories/levels to run (e.g. level1,math)',
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help='Model config name from configs.toml (e.g. qwen3-coder-next)',
    )
    parser.add_argument(
        '--skip-hardware-probe',
        action='store_true',
        help='Do not call bench_api POST /baseline at startup (skip GPU name for prompts). '
        'Or set env KERNEL_AGENT_SKIP_HARDWARE_PROBE=1.',
    )
    args = parser.parse_args()

    cfg = load_config(args.model)
    search_cfg = load_search_config(args.search_method)
    stage_cfg = load_stage_config()
    global MODEL_NAME, VLLM_BASE_URL
    MODEL_NAME    = cfg['model_name']
    VLLM_BASE_URL = cfg['base_url']

    example_pre_path = Path('examples/cuda_model_add.py')
    example_pre = example_pre_path.read_text()

    example_after_path = Path('examples/cuda_new_model_add.py')
    example_after = example_after_path.read_text()

    bench_url = os.getenv('BENCHAPI_URL', 'http://localhost:8123').rstrip('/')
    print(f'[DEBUG] BENCHAPI_URL={bench_url}', flush=True)

    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as httpx_client:
        skip_hw = args.skip_hardware_probe or os.getenv(
            'KERNEL_AGENT_SKIP_HARDWARE_PROBE', ''
        ).lower() in ('1', 'true', 'yes')
        if skip_hw:
            hardware_info = ''
            compute_capability = None
            print('[DEBUG] Skipping bench_api /baseline hardware probe (--skip-hardware-probe or KERNEL_AGENT_SKIP_HARDWARE_PROBE).')
            print('Target Hardware: Not specified')
        else:
            print('[DEBUG] Getting hardware info...')
            hw_data = await get_hardware_info(client=httpx_client)
            hardware_info = hw_data['hardware'] or ''
            compute_capability = hw_data['compute_capability']
            if hardware_info:
                hw_str = f"{hardware_info}"
                if compute_capability:
                    hw_str += f" (Compute Capability: {compute_capability})"
                print(f'Target Hardware: {hw_str}')
            else:
                print('Target Hardware: Not specified')

        print(f'[DEBUG] Discovering kernels from {args.dataset}...')
        if args.dataset == 'kernelbench':
            base_path = Path('../KernelBench/KernelBench')
            # Discover all level* directories
            all_reference_files = []
            for level_dir in sorted(base_path.glob('level*')):
                if level_dir.is_dir():
                    all_reference_files.extend(list(level_dir.glob('*.py')))
        elif args.dataset == 'multikernelbench':
            base_path = Path('../MultiKernelBench/reference')
            all_reference_files = list(base_path.glob('**/*.py'))
        else:
            base_path = Path(args.dataset)
            all_reference_files = list(base_path.glob('**/*.py'))

        reference_files = []
        for f in all_reference_files:
            if args.categories:
                target_categories = set(args.categories.split(','))
                if f.parent.name not in target_categories:
                    continue
            if args.kernels:
                target_kernels = set(args.kernels.split(','))
                if f.stem not in target_kernels:
                    continue
            reference_files.append(f)

        if not reference_files:
            print('Warning: No kernels found matching criteria.')

        reference_files = sorted(reference_files)
        from collections import defaultdict
        cat_to_files = defaultdict(list)
        for f in reference_files:
            cat_to_files[f.parent.name].append(f)
        reference_files = []
        for cat in sorted(cat_to_files.keys()):
            files = cat_to_files[cat]
            random.Random(GLOBAL_SEED).shuffle(files)
            reference_files.extend(files)

        print(f'Found {len(reference_files)} kernels to process.')
        print('[DEBUG] Starting processing loop...')
        global experiment_logs, VLLM_SUPPORTS_N_SAMPLING, GEN_SEMAPHORE, EVAL_SEMAPHORE, KERNEL_SEMAPHORE
        client = AsyncOpenAI(api_key=cfg['api_key'], base_url=VLLM_BASE_URL)
        VLLM_SUPPORTS_N_SAMPLING = (
            await probe_n_sampling(client, MODEL_NAME, n=search_cfg['width'])
            if search_cfg['width'] > 1 else True
        )

        pk = args.parallel_kernels
        GEN_SEMAPHORE = asyncio.Semaphore(16 * pk)
        EVAL_SEMAPHORE = asyncio.Semaphore(8 * pk)
        KERNEL_SEMAPHORE = asyncio.Semaphore(pk)

        try:
            run_idx = 0
            experiment_logs = []
            run_file = Path(f'data/{args.exp_id}/results.parquet')
            completed_tasks = set()
            if args.search_method == 'sampling':
                # batch5 + 5 advice rows + ≥1 sixth row (early exit when sixth is valid)
                expected_nodes_per_kernel = (
                    search_cfg['width'] * search_cfg['height'] + 5 + 1
                )
            else:
                expected_nodes_per_kernel = search_cfg['width'] * search_cfg['height']

            if run_file.exists():
                try:
                    df = pd.read_parquet(run_file)
                    # Count records per (kernel_name, search_method)
                    counts = df.groupby(['kernel_name', 'search_method']).size()
                    completed_tasks = set()
                    partial_tasks = set()
                    for (kname, smethod), cnt in counts.items():
                        if cnt >= expected_nodes_per_kernel:
                            completed_tasks.add((kname, smethod))
                        else:
                            partial_tasks.add((kname, smethod))
                    # Kernels that have all expected records but some failed with
                    # a network error — re-run them so those nodes get real results.
                    error_tasks = set(
                        (r['kernel_name'], r['search_method'])
                        for r in df.to_dict('records')
                        if isinstance(r.get('compile_info'), str)
                        and r['compile_info'].startswith('bench_api_error')
                    )
                    # Move error kernels from completed → partial
                    partial_tasks |= error_tasks
                    completed_tasks -= error_tasks
                    # Drop all records for partial/error kernels — regenerate from scratch
                    if partial_tasks:
                        keep_mask = ~df.apply(
                            lambda r: (r['kernel_name'], r['search_method']) in partial_tasks,
                            axis=1,
                        )
                        df = df[keep_mask]
                        print(
                            f'Dropped {(~keep_mask).sum()} records for '
                            f'{len(partial_tasks)} kernel(s) to re-run '
                            f'({len(error_tasks)} had bench_api_error): '
                            f'{sorted(k for k, _ in partial_tasks)}'
                        )
                    experiment_logs = df.to_dict('records')
                    print(
                        f'Resume: loaded {len(experiment_logs)} records, '
                        f'{len(completed_tasks)} complete kernels skipped, '
                        f'{len(partial_tasks)} kernel(s) will be re-run.'
                    )
                except Exception as e:
                    print(f'Error reading existing log file {run_file}: {e}')

            in_progress = set()

            async def process_kernel(file_idx, target_path):
                function_name = target_path.stem
                try:
                    async with KERNEL_SEMAPHORE:
                        print(f'\n[{file_idx + 1}/{len(reference_files)}] {target_path} | Experiment {args.exp_id}')
                        kernel_category = target_path.parent.name
                        reference_code = target_path.read_text()
                        if args.search_method == 'sampling':
                            await sampling_feedback_sixth_search(
                                client=client,
                                model_name=MODEL_NAME,
                                reference_code=reference_code,
                                hardware_info=hardware_info,
                                compute_capability=compute_capability,
                                example_pre=example_pre,
                                example_after=example_after,
                                language='cuda',
                                function_name=function_name,
                                run_idx=run_idx,
                                run_seed=GLOBAL_SEED,
                                width=search_cfg['width'],
                                height=search_cfg['height'],
                                selector=GreedySelector(beam_size=search_cfg['beam_size']),
                                kernel_category=kernel_category,
                                dataset=args.dataset,
                                search_method_name=args.search_method,
                                temperature=search_cfg['temperature'],
                                top_p=search_cfg['top_p'],
                                httpx_client=httpx_client,
                                diagnoser_temperature=stage_cfg['diagnoser_temperature'],
                                advisor_temperature=stage_cfg['advisor_temperature'],
                                diagnoser_max_tokens=stage_cfg['diagnoser_max_tokens'],
                                advisor_max_tokens=stage_cfg['advisor_max_tokens'],
                                coder_max_tokens=stage_cfg['coder_max_tokens'],
                                sixth_max_attempts=int(search_cfg.get('sixth_max_attempts', 3)),
                                advice_temperature=float(
                                    search_cfg.get('advice_temperature', stage_cfg['advisor_temperature'])
                                ),
                                advice_max_tokens=int(search_cfg.get('advice_max_tokens', 160)),
                            )
                        else:
                            await tree_search(
                                client=client,
                                model_name=MODEL_NAME,
                                reference_code=reference_code,
                                hardware_info=hardware_info,
                                compute_capability=compute_capability,
                                example_pre=example_pre,
                                example_after=example_after,
                                language='cuda',
                                function_name=function_name,
                                run_idx=run_idx,
                                run_seed=GLOBAL_SEED,
                                width=search_cfg['width'],
                                height=search_cfg['height'],
                                selector=GreedySelector(beam_size=search_cfg['beam_size']),
                                kernel_category=kernel_category,
                                dataset=args.dataset,
                                search_method_name=args.search_method,
                                temperature=search_cfg['temperature'],
                                top_p=search_cfg['top_p'],
                                httpx_client=httpx_client,
                                two_stage=search_cfg.get('two_stage', True),
                                diagnoser_temperature=stage_cfg['diagnoser_temperature'],
                                advisor_temperature=stage_cfg['advisor_temperature'],
                                diagnoser_max_tokens=stage_cfg['diagnoser_max_tokens'],
                                advisor_max_tokens=stage_cfg['advisor_max_tokens'],
                                coder_max_tokens=stage_cfg['coder_max_tokens'],
                            )
                except Exception as e:
                    print(f'[{function_name}] process_kernel failed: {type(e).__name__}: {e}', flush=True)
                finally:
                    async with SAVE_LOCK:
                        save_logs(args.exp_id)

            tasks = []
            for file_idx, target_path in enumerate(reference_files):
                function_name = target_path.stem
                key = (function_name, args.search_method)
                if key in completed_tasks or key in in_progress:
                    print(f'Skipping {function_name} / {args.search_method} - already completed.')
                    continue
                in_progress.add(key)
                tasks.append(process_kernel(file_idx, target_path))

            gather_results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, res in enumerate(gather_results):
                if isinstance(res, BaseException):
                    print(f'[gather] Task {i} raised: {type(res).__name__}: {res}', flush=True)
            async with SAVE_LOCK:
                save_logs(args.exp_id)

        except Exception as e:
            print(f'Error: {e}')
            raise


if __name__ == '__main__':
    asyncio.run(main())
