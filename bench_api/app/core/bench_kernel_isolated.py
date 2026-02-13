import subprocess
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

ISOLATED_SCRIPT = Path(__file__).parent.parent.parent / 'eval_single_isolated.py'


def evaluate_kernel_isolated(
    function_code: str,
    function: str,
    language: str,
    torch_compile: bool = False,
    num_trials: Optional[int] = None,
    batch_size: Optional[int] = None,
    dim: Optional[int] = None,
    input_dims: Optional[Dict[str, Any]] = None,
    timeout: int = 300
) -> Dict[str, Any]:
    params = {
        'function_code': function_code,
        'function': function,
        'language': language,
        'torch_compile': torch_compile,
        'num_trials': num_trials,
        'batch_size': batch_size,
        'dim': dim,
        'input_dims': input_dims
    }

    try:
        result = subprocess.run(
            ['python3', str(ISOLATED_SCRIPT)],
            input=json.dumps(params),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ISOLATED_SCRIPT.parent.parent)
        )

        print(result.returncode)


        stdout_text = result.stdout.strip() if result.stdout else ""
        stderr_text = result.stderr.strip() if result.stderr else ""
        
        json_output = None
        if stderr_text:
            lines = stderr_text.split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line and (line.startswith('{') or line.startswith('[')):
                    try:
                        json_output = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
        
        if json_output is None and stdout_text:
            lines = stdout_text.split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line and (line.startswith('{') or line.startswith('[')):
                    try:
                        json_output = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
        
        if json_output is not None:
            return json_output
        
        error_msg = stderr_text or stdout_text or "Unknown error"
        return {
            'compiled': False,
            'correctness': None,
            'performance': None,
            'hardware': 'unknown',
            'error': f"Failed to find JSON in subprocess output (returncode={result.returncode}): {error_msg[:1000]}"
        }

    except subprocess.TimeoutExpired:
        return {
            'compiled': False,
            'correctness': None,
            'performance': None,
            'hardware': 'unknown',
            'error': f"Evaluation timed out after {timeout} seconds"
        }
    except Exception as e:
        return {
            'compiled': False,
            'correctness': None,
            'performance': None,
            'hardware': 'unknown',
            'error': f"Subprocess execution failed: {str(e)}"
        }
