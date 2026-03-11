"""
Isolated evaluation worker.

Invoked as:  python -m app.core.eval_worker   (cwd = bench_api/)
Reads JSON params from stdin, writes a single JSON result line to stdout.
stderr is for logs only.
"""
import json
import os
import signal
import sys


def _write(result: dict) -> None:
    sys.stdout.write(json.dumps(result) + '\n')
    sys.stdout.flush()


def _signal_handler(signum, frame):
    try:
        from app.core.bench_kernel import get_current_eval_stage, EvalStage
        stage = get_current_eval_stage()
    except Exception:
        stage = 'initialization'

    sig_name = signal.Signals(signum).name
    error_msg = f"Process received signal {signum} ({sig_name}) during {stage} stage"
    result = {
        'compiled': stage not in ['initialization', 'baseline', 'compilation'],
        'correctness': False if stage == 'correctness' else None,
        'performance': None,
        'hardware': 'unknown',
        'stage': stage,
        'error': error_msg,
    }
    if stage == 'compilation':
        result['compile_info'] = error_msg
    elif stage == 'correctness':
        result['correctness_info'] = error_msg
    elif stage == 'performance':
        result['performance_error'] = error_msg

    _write(result)
    os._exit(signum)


def main():
    for sig in [signal.SIGSEGV, signal.SIGFPE, signal.SIGABRT, signal.SIGBUS]:
        signal.signal(sig, _signal_handler)

    params = json.loads(sys.stdin.read())
    mode = params.get('mode', 'validation')

    if mode != 'benchmark':
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    from app.integration import register_kernelbench_dataset
    register_kernelbench_dataset()

    from app.core.bench_kernel import evaluate_kernel

    try:
        result = evaluate_kernel(
            function_code=params['function_code'],
            function=params['function'],
            language=params['language'],
            torch_compile=params.get('torch_compile', False),
            torch_compile_baseline=params.get('torch_compile_baseline', False),
            num_trials=params.get('num_trials'),
            num_warmup=params.get('num_warmup'),
            batch_size=params.get('batch_size'),
            dim=params.get('dim'),
            input_dims=params.get('input_dims'),
            mode=mode,
            baseline_mean_ms=params.get('baseline_mean_ms'),
        )
    except Exception as e:
        import traceback
        result = {
            'compiled': False,
            'correctness': None,
            'performance': None,
            'hardware': 'unknown',
            'error': f"Evaluation failed: {e}\n{traceback.format_exc()}",
        }

    _write(result)


if __name__ == '__main__':
    main()
