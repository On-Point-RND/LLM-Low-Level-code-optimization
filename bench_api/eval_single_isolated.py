import sys
import json
import traceback

def run_isolated_evaluation():
    try:
        params = json.loads(sys.stdin.read())
        
        from app.core.bench_kernel import evaluate_kernel as evaluate_kernel_impl
        
        result = evaluate_kernel_impl(
            function_code=params['function_code'],
            function=params['function'],
            language=params['language'],
            torch_compile=params.get('torch_compile', False),
            num_trials=params.get('num_trials'),
            batch_size=params.get('batch_size'),
            dim=params.get('dim'),
            input_dims=params.get('input_dims')
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
