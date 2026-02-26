import sys
import json
import traceback
import signal
import os

def signal_handler(signum, frame):
    from app.core.bench_kernel import get_current_eval_stage, EvalStage
    
    stage = get_current_eval_stage()
    sig_name = signal.Signals(signum).name
    error_msg = f"Process received signal {signum} ({sig_name}) during {stage} stage"
    
    result = {
        'compiled': stage not in [EvalStage.INITIALIZATION, EvalStage.BASELINE, EvalStage.COMPILATION],
        'correctness': False if stage == EvalStage.CORRECTNESS else None,
        'performance': None,
        'hardware': 'unknown',
        'stage': stage,
        'error': error_msg
    }
    
    if stage == EvalStage.COMPILATION:
        result['compile_info'] = error_msg
    elif stage == EvalStage.CORRECTNESS:
        result['correctness_info'] = error_msg
    elif stage == EvalStage.PERFORMANCE:
        result['performance_error'] = error_msg
        
    sys.stderr.write(json.dumps(result) + '\n')
    sys.stderr.flush()
    os._exit(signum)

def run_isolated_evaluation():
    for sig in [signal.SIGSEGV, signal.SIGFPE, signal.SIGABRT, signal.SIGBUS]:
        signal.signal(sig, signal_handler)

    try:
        params = json.loads(sys.stdin.read())

        mode = params.get('mode', 'validation')

        if mode != 'benchmark':
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        from app.integration import register_kernelbench_dataset
        register_kernelbench_dataset()

        from app.core.bench_kernel import evaluate_kernel as evaluate_kernel_impl

        result = evaluate_kernel_impl(
            function_code=params['function_code'],
            function=params['function'],
            language=params['language'],
            torch_compile=params.get('torch_compile', False),
            num_trials=params.get('num_trials'),
            batch_size=params.get('batch_size'),
            dim=params.get('dim'),
            input_dims=params.get('input_dims'),
            mode=mode,
            baseline_mean_ms=params.get('baseline_mean_ms'),
        )
        
        sys.stderr.write(json.dumps(result) + '\n')
        sys.exit(0)
        
    except Exception as e:
        error_result = {
            'compiled': False,
            'correctness': None,
            'performance': None,
            'hardware': 'unknown',
            'error': f"Isolated evaluation failed: {str(e)}\n{traceback.format_exc()}"
        }
        sys.stderr.write(json.dumps(error_result) + '\n')
        sys.exit(1)

if __name__ == '__main__':
    try:
        run_isolated_evaluation()
    except Exception as e:
        error_result = {
            'compiled': False,
            'correctness': None,
            'performance': None,
            'hardware': 'unknown',
            'error': f"Fatal error in isolated evaluation script: {str(e)}\n{traceback.format_exc()}"
        }
        sys.stderr.write(json.dumps(error_result) + '\n')
        sys.exit(1)
