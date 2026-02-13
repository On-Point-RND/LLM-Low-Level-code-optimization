# eval_single_runner.py
import sys
import json
from utils.evaluation_utils import eval_single  

response_txt_path = sys.argv[1]
op = sys.argv[2]
language = sys.argv[3]
result_path = sys.argv[4]

with open(response_txt_path, 'r') as f:
    response_txt = f.read()

result = eval_single(response_txt, op, language)

with open(result_path, 'w') as f:
    json.dump(result, f)
        