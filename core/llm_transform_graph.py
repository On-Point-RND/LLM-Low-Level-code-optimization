import os
import json
import tvm
from .schemas import (
    OpenRouterConfig,
    OpenRouterRequest,
    OpenRouterMessage
)

def _load_graph_prompt():
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "tvm_graph_prompt.txt")
    with open(prompt_path, "r") as f:
        return f.read()

def _create_openrouter_client(config: OpenRouterConfig):
    import urllib.request
    
    def call_llm(content: str) -> dict:
        graph_prompt = _load_graph_prompt()
        
        system_prompt = f"""{graph_prompt}

Your task is to optimize the TVM Relax JSON graph representation.
- Optimize the graph structure by replacing operations with more efficient alternatives
- Maintain all required fields and structure
- Preserve functional correctness
- Return only the optimized JSON graph, no explanations or additional text"""

        user_prompt = f"""Optimize the following TVM Relax JSON graph:

{content}

Return only the optimized JSON graph, no additional text."""

        messages = [
            OpenRouterMessage(role="system", content=system_prompt),
            OpenRouterMessage(role="user", content=user_prompt)
        ]
        
        openrouter_req = OpenRouterRequest(
            model=config.default_model,
            messages=messages,
            temperature=config.default_temperature,
            max_tokens=config.default_max_tokens
        )
        
        url = f"{config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/tvm",
            "X-Title": "TVM Benchmark"
        }
        
        request_data = openrouter_req.model_dump(exclude_none=True)
        data = json.dumps(request_data).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        
        with urllib.request.urlopen(req, timeout=config.timeout) as response:
            response_data = json.loads(response.read().decode('utf-8'))
        
        if "choices" not in response_data or len(response_data["choices"]) == 0:
            raise RuntimeError(f"Invalid response from OpenRouter: {response_data}")
        
        transformed_content = response_data["choices"][0]["message"]["content"]
        
        return {
            "transformed_content": transformed_content,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "original_content": content,
            "llm_response": response_data,
            "request_data": request_data,
            "model": config.default_model,
            "url": url
        }
    
    return call_llm

def llm_transform_graph(mod, config: OpenRouterConfig):
    graph_json_obj = tvm.ir.save_json(mod)
    graph_json = str(graph_json_obj)
    
    call_llm = _create_openrouter_client(config)
    
    print("      Sending graph JSON to LLM for transformation...")
    llm_result = call_llm(graph_json)
    transformed_json = llm_result["transformed_content"]
    
    print("      Loading transformed graph...")
    error_msg = None
    transformed_mod = None
    
    try:
        transformed_mod = tvm.ir.load_json(transformed_json)
    except Exception as e:
        error_msg = str(e)
        print(f"      ERROR: Failed to load JSON graph: {error_msg}")
        print("      Using original module")
    
    if transformed_mod is None:
        llm_result["load_error"] = error_msg or "Unknown error"
        return mod, llm_result
    
    return transformed_mod, llm_result
