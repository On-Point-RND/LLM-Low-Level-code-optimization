import os
import json
import tvm
from .schemas import (
    OpenRouterConfig,
    OpenRouterRequest,
    OpenRouterMessage,
)
import urllib.request

def _load_tir_script_prompt():
    prompt_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "prompts",
        "tvm_tir_prompt.txt",
    )
    with open(prompt_path, "r") as f:
        return f.read()


def _create_openrouter_client(config: OpenRouterConfig):

    def call_llm(content: str) -> dict:
        script_prompt = _load_tir_script_prompt()

        system_prompt = f"""{script_prompt}

Your task is to optimize the low-level TVM TIR / TVMScript code (schedules and memory accesses).
- Focus on loop transformations (tiling, fusion, reordering), vectorization and memory hierarchy usage
- Maintain functional correctness and all type annotations
- Preserve function signatures and attributes
- Do NOT invent new TVM intrinsics, attributes or helper variables (including any `metadata[...]` access);
  use only valid TIR/TVMScript constructs and symbols present in the input code.
- You MUST respond with a single JSON object with the following shape:
  {{
    "code": "string - optimized TVM TIR / TVMScript code",
    "status": "string - 'improved' or 'original'",
    "modification": "string - short description of what was changed"
  }}
- No additional fields or text outside this JSON."""

        user_prompt = f"""Optimize the following TVM TIR / TVMScript code:

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
                "name": "tvm_tir_transform",
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

        if "choices" not in response_data or not response_data["choices"]:
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


def llm_transform_tir(mod, config: OpenRouterConfig):
    ir_script = mod.script()

    call_llm = _create_openrouter_client(config)

    print("      Sending TIR / TVMScript to LLM for scheduling...")
    llm_result = call_llm(ir_script)
    raw_response = llm_result["transformed_content"]


    payload = json.loads(raw_response)


    transformed_script = payload.get("code")
    status = payload.get("status")
    modification = payload.get("modification")

    if not isinstance(transformed_script, str) or not transformed_script.strip():
        msg = "LLM structured output (TIR) missing non-empty 'code' field"
        llm_result["parse_error"] = msg
        return None, llm_result

    llm_result["structured_output"] = {
        "code": transformed_script,
        "status": status,
        "modification": modification,
    }

    print("      Parsing transformed TIR / TVMScript...")
    error_msg = None
    transformed_mod = None

    parsed = tvm.script.from_source(transformed_script)
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

