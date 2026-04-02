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


def load_config(name: str) -> dict:
    with open('configs.toml', 'rb') as f:
        all_configs = tomllib.load(f)
    if name not in all_configs:
        raise ValueError(f"Config '{name}' not found in configs.toml. Available: {list(all_configs)}")
    cfg = all_configs[name]
    if 'api_key' not in cfg:
        cfg['api_key'] = os.environ['OPENAI_API_KEY']
    return cfg


def load_search_config(method: str) -> dict:
    with open('configs.toml', 'rb') as f:
        all_configs = tomllib.load(f)
    if method not in all_configs:
        raise ValueError(f"Search method '{method}' not found in configs.toml. Available: {list(all_configs)}")
    return all_configs[method]

experiment_logs = []

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
        async with httpx.AsyncClient(timeout=4800.0) as new_client:
            response = await new_client.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
    return {
        'hardware': data.get('hardware'),
        'compute_capability': data.get('compute_capability'),
    }


async def evaluate_kernel(code: str, language: str, function_name: str, client: httpx.AsyncClient | None = None) -> dict:
    url = os.getenv('BENCHAPI_URL', 'http://localhost:8123') + '/evaluate'
    payload = {
        'language': language,
        'function': function_name,
        'function_code': code,
        'torch_compile_baseline': True,
    }
    if client:
        response = await client.post(url, json=payload, timeout=4800.0)
    else:
        async with httpx.AsyncClient(timeout=4800.0) as new_client:
            response = await new_client.post(url, json=payload, timeout=4800.0)
    response.raise_for_status()
    return response.json()


def _jinja_env() -> Environment:
    return Environment(loader=FileSystemLoader('prompts'))


def generation_prompt(
    reference_code: str,
    hardware_info: str,
    compute_capability: str | None = None,
    example_pre: str | None = None,
    example_after: str | None = None,
    last_code: str | None = None,
    last_result: dict | None = None,
    history: list[dict] | None = None,
) -> str:
    template = _jinja_env().get_template('kernel_optimization.md.jinja')
    return template.render(
        reference_code=reference_code,
        hardware_info=hardware_info,
        compute_capability=compute_capability,
        example_pre=example_pre,
        example_after=example_after,
        last_code=last_code,
        last_result=last_result,
        history=history or [],
    )


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


def extract_code_from_response(response: str) -> str:
    if not response:
        return ''
    trimmed = response.strip()
    if not trimmed:
        return ''
    code_match = re.search(r'```(.*?)```', trimmed, re.DOTALL)
    if code_match:
        code_block = code_match.group(1).strip()
        for code_type in ['python', 'cpp', 'cuda', 'triton']:
            if code_block.startswith(code_type):
                code_block = code_block[len(code_type) :].strip()
        return code_block
    return trimmed


async def _api_call_with_retry(coro_fn, max_retries: int = 10, retry_delay: float = 5.0):
    """Retry an OpenAI API call on connection errors with constant delay."""
    for attempt in range(max_retries):
        try:
            response = await coro_fn()
            if response is None or response.choices is None:
                raise openai.APIConnectionError(request=None)
            return response
        except openai.APIConnectionError as e:
            if attempt == max_retries - 1:
                raise
            print(f'[retry] Connection error (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay:.0f}s: {e}')
            await asyncio.sleep(retry_delay)


async def _generate_code(
    client: AsyncOpenAI,
    messages: list[dict],
    model_name: str,
    seed: int,
    temperature: float = 0.7,
    top_p: float = 0.8,
    n: int = 1,
    extra_body: dict | None = None,
) -> tuple[list[str], list[str], any]:
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
                    max_tokens=16384,
                    n=1,
                    extra_body=extra_body,
                ))
        responses = await asyncio.gather(*[_single(i) for i in range(n)])
        raw_responses = [r.choices[0].message.content for r in responses]
        codes = [extract_code_from_response(r) for r in raw_responses]
        return codes, raw_responses, responses[0].usage

    for _ in range(n):
        await GEN_SEMAPHORE.acquire()
    try:
        response = await _api_call_with_retry(lambda: client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_tokens=16384,
            n=n,
            extra_body=extra_body,
        ))
    finally:
        for _ in range(n):
            GEN_SEMAPHORE.release()

    raw_responses = [choice.message.content for choice in response.choices]
    codes = [extract_code_from_response(r) for r in raw_responses]
    return codes, raw_responses, response.usage


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
) -> dict:
    async with EVAL_SEMAPHORE:
        t_eval_start = time.time()
        eval_result = await evaluate_kernel(generated_code, language, function_name, client=httpx_client)
        t_eval_end = time.time()

    compiled = eval_result.get('compiled', False)
    correctness = eval_result.get('correctness', False)
    speedup = eval_result.get('speedup')
    compilation_message = eval_result.get('compile_info')
    correctness_info = eval_result.get('correctness_info')
    performance_info = eval_result.get('performance_info')
    timing = eval_result.get('timing') or {}

    print(
        f'[{function_name}] Node {node_id}: Compiled: {compiled}, Valid: {correctness}, Speedup: {f"{speedup:.2f}x" if speedup else "N/A"} | {hypothesis or "(no hypothesis)"}'
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
            'duration_total': t_eval_end - t_gen_start,
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
            'speedup': speedup,
            'compilation_message': compilation_message,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cached_tokens': cached_tokens,
            'node_id': node_id,
            'parent_node_id': parent_node_id,
            'expansion_id': expansion_id,
            'hypothesis': hypothesis,
        }
    )

    return {
        'code': generated_code,
        'compiled': compiled,
        'compilation_error': compilation_message,
        'valid': correctness,
        'correctness_info': correctness_info,
        'speedup': speedup,
        'performance_info': performance_info,
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
                return (2, r.get('speedup') or 0.0)
            elif r.get('compiled'):
                return (1, 0.0)
            return (0, 0.0)

        return sorted(candidates, key=score, reverse=True)[: self.beam_size]


def _collect_ancestry(node: TreeNode) -> tuple[str | None, dict | None, list[dict]]:
    """
    Build prompt context from a node's ancestry chain.

    Returns (last_code, last_result, history) where:
    - last_code / last_result: node's own code and evaluation result
      (None if node is root — i.e. this is the first attempt)
    - history: compact list of {hypothesis, result} for all older ancestors,
      oldest first (empty for depth-0 and depth-1 nodes)
    """
    if node.parent is None:
        # node is root — no prior attempts
        return None, None, []

    # Collect path from root's child down to node (inclusive)
    path: list[TreeNode] = []
    current = node
    while current.parent is not None:
        path.append(current)
        current = current.parent
    path.reverse()  # oldest first

    last = path[-1]
    history = [{'hypothesis': n.hypothesis, 'result': n.result} for n in path[:-1]]
    return last.code, last.result, history


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
):
    """
    Generic tree search over kernel generations.

    width=1, height=N  -> iterative refinement (single path, sequential feedback)
    width=N, height=1  -> multiple sampling (N parallel candidates, no feedback)
    width=W, height=H  -> beam / tree search (W children per node, H levels deep)

    At each level the selector chooses which evaluated nodes to expand next.
    """
    print(f'\n=== {search_method_name.replace("_", " ").title()}: Run {run_idx + 1} (Seed: {run_seed}) ===')

    root = TreeNode(messages=[], depth=0)
    frontier = [root]
    eval_counter = 0
    expansion_counter = 0

    for depth in range(height):
        print(f'--- Depth {depth + 1}/{height}, frontier size: {len(frontier)}, branching x{width} ---')

        async def process_parent(parent, node_expansion_id, node_eval_id_start):
            last_code, last_result, history = _collect_ancestry(parent)
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant'},
                {
                    'role': 'user',
                    'content': generation_prompt(
                        reference_code=reference_code,
                        hardware_info=hardware_info,
                        compute_capability=compute_capability,
                        example_pre=example_pre,
                        example_after=example_after,
                        last_code=last_code,
                        last_result=last_result,
                        history=history,
                    ),
                },
            ]

            # Vary seed per expansion call so repeated expansion of the same node
            # (exploitation) produces diverse children rather than identical copies.
            call_seed = (run_seed + node_expansion_id) % (2**32)

            t_gen_start = time.time()
            codes, raw_responses, usage = await _generate_code(
                client=client,
                messages=messages,
                model_name=model_name,
                seed=call_seed,
                temperature=temperature,
                top_p=top_p,
                n=width,
                extra_body={'cache_salt': f'{function_name}-{run_seed}'},
            )
            t_gen_end = time.time()

            u = usage if isinstance(usage, dict) else (usage.model_dump() if usage else {})
            prompt_tokens = u.get('prompt_tokens')
            completion_tokens = u.get('completion_tokens')
            cached_tokens = (u.get('prompt_tokens_details') or {}).get('cached_tokens')

            if prompt_tokens is None or completion_tokens is None:
                raise ValueError(f'Critical: Missing token info in usage: {u}')

            eval_tasks = []
            for i, (code, raw) in enumerate(zip(codes, raw_responses)):
                # print(code)
                hypothesis = parse_hypothesis(code)
                eval_tasks.append(
                    _evaluate_and_log(
                        generated_code=code,
                        prompt_messages=messages,
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
                    )
                )

            eval_results = await asyncio.gather(*eval_tasks)
            
            children = []
            for i, (result, code, raw) in enumerate(zip(eval_results, codes, raw_responses)):
                child = TreeNode(
                    messages=messages,
                    depth=depth + 1,
                    parent=parent,
                    node_id=node_eval_id_start + i,
                    code=code,
                    raw_response=raw,
                    hypothesis=parse_hypothesis(code),
                    result=result,
                )
                children.append(child)
            return children

        expansion_tasks = []
        for parent in frontier:
            expansion_tasks.append(process_parent(parent, expansion_counter, eval_counter))
            expansion_counter += 1
            eval_counter += width

        results = await asyncio.gather(*expansion_tasks)
        next_candidates = [child for branch in results for child in branch]

        if depth < height - 1:
            selected = selector.select(next_candidates)
            if not selected:
                print('Selector returned empty frontier, stopping early.')
                break
            frontier = selected




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
        choices=['kernelbench', 'multikernelbench'],
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
    args = parser.parse_args()

    cfg = load_config(args.model)
    search_cfg = load_search_config(args.search_method)
    global MODEL_NAME, VLLM_BASE_URL
    MODEL_NAME    = cfg['model_name']
    VLLM_BASE_URL = cfg['base_url']

    example_pre_path = Path('examples/cuda_model_add.py')
    example_pre = example_pre_path.read_text()

    example_after_path = Path('examples/cuda_new_model_add.py')
    example_after = example_after_path.read_text()

    async with httpx.AsyncClient(timeout=4800.0) as httpx_client:
        print('[DEBUG] Getting hardware info...')
        hw_data = await get_hardware_info(client=httpx_client)
        hardware_info = hw_data['hardware']
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
        else:
            base_path = Path('../MultiKernelBench/reference')
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
            if run_file.exists():
                try:
                    df = pd.read_parquet(run_file)
                    experiment_logs = df.to_dict('records')
                    completed_tasks = set(zip(df['kernel_name'], df['search_method']))
                    print(f'Loaded {len(experiment_logs)} existing records for Experiment {args.exp_id}')
                except Exception as e:
                    print(f'Error reading existing log file {run_file}: {e}')

            in_progress = set()

            async def process_kernel(file_idx, target_path):
                function_name = target_path.stem
                async with KERNEL_SEMAPHORE:
                    print(f'\n[{file_idx + 1}/{len(reference_files)}] {target_path} | Experiment {args.exp_id}')
                    kernel_category = target_path.parent.name
                    reference_code = target_path.read_text()
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
                    )
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

            await asyncio.gather(*tasks)
            async with SAVE_LOCK:
                save_logs(args.exp_id)

        except Exception as e:
            print(f'Error: {e}')
            raise


if __name__ == '__main__':
    asyncio.run(main())
