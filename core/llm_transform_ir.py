import os
import json
import tvm
from .schemas import (
    OpenRouterConfig,
    OpenRouterRequest,
    OpenRouterMessage,
)

def _load_script_prompt():
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "tvm_script_prompt.txt")
    with open(prompt_path, "r") as f:
        return f.read()


def _extract_code_block(raw: str) -> str:
    start_token = "##@@##"
    end_token = "##%%##"
    start = raw.find(start_token)
    end = raw.rfind(end_token)
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError(
            "LLM response does not contain a valid code block "
            "delimited by ##@@## (start) and ##%%## (end)."
        )
    return raw[start + len(start_token) : end].strip()

def _create_openrouter_client(config: OpenRouterConfig):
    import urllib.request

    def call_llm(content: str) -> dict:
        script_prompt = _load_script_prompt()

        system_prompt = f"""{script_prompt}

Your task is to optimize the TVM Relax TVMScript code.
- Optimize the computation graph by replacing operations with more efficient alternatives
- Maintain functional correctness
- Preserve all type annotations and function signatures
- Do NOT use metadata or metadata[...] (e.g. metadata["relax.expr.Constant"][0]) — it is undefined in TVMScript and causes parse errors. Weights/constants must stay as the same parameter names as in the input (e.g. input_0, input_1).
- Do NOT invent new TVM operators, attributes, or helper variables; use only valid TVM Relax constructs and symbols present in the input code.
- You MUST respond with a single JSON object with the following shape:
  {{
    "code": "string - optimized TVM Relax TVMScript code",
    "status": "string - 'improved' or 'original'",
    "modification": "string - short description of what was changed"
  }}
- No additional fields or text outside this JSON."""

        user_prompt = f"""Optimize the following TVM Relax TVMScript code:

{content}

Return ONLY a JSON object with fields "code", "status", "modification" as described above."""

        messages = [
            OpenRouterMessage(role="system", content=system_prompt),
            OpenRouterMessage(role="user", content=user_prompt),
        ]

        schema = {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "status": {"type": "string", "enum": ["improved", "original"]},
                "modification": {"type": "string"},
            },
            "required": ["code", "status", "modification"],
            "additionalProperties": False,
        }

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "tvm_relax_transform",
                "strict": True,
                "schema": schema,
            },
        }

        openrouter_req = OpenRouterRequest(
            model=config.default_model,
            messages=messages,
            temperature=config.default_temperature,
            max_tokens=config.default_max_tokens,
            response_format=response_format,
        )

        url = f"{config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/tvm",
            "X-Title": "TVM Benchmark",
        }

        request_data = openrouter_req.model_dump(exclude_none=True)
        data = json.dumps(request_data).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=config.timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))

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
            "url": url,
        }

    return call_llm

def llm_transform_ir(mod, config: OpenRouterConfig):
    ir_script = mod.script()

    call_llm = _create_openrouter_client(config)

    print("      Sending IR to LLM for transformation...")
    llm_result = call_llm(ir_script)
    raw_response = llm_result["transformed_content"]

    # Strict structured output: parse JSON with fields code/status/modification
    payload = json.loads(raw_response)


    code_block = payload.get("code")
    status = payload.get("status")
    modification = payload.get("modification")

    if not isinstance(code_block, str) or not code_block.strip():
        msg = "LLM structured output (IR) missing non-empty 'code' field"
        llm_result["parse_error"] = msg
        return None, llm_result

    llm_result["structured_output"] = {
        "code": code_block,
        "status": status,
        "modification": modification,
    }

    print("      Parsing transformed IR...")
    error_msg = None
    transformed_mod = None
    parsed = tvm.script.from_source(code_block)
    if isinstance(parsed, tvm.ir.IRModule):
        transformed_mod = parsed
    else:
        error_msg = f"Parsed result is not an IRModule, got {type(parsed)}"
        print(f"      ERROR: {error_msg}")


    if transformed_mod is None:
        msg = error_msg or "Unknown error"
        llm_result["parse_error"] = msg
        # Вернём None, чтобы пайплайн мог залогировать llm_log и упасть выше
        return None, llm_result
    return transformed_mod, llm_result
