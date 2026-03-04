from typing import Literal
from pydantic import BaseModel

from ..schemas import OpenRouterConfig, OpenRouterMessage
from .base import load_prompt_config, load_tir_syntax, call_openrouter


class TIRTransformResponse(BaseModel):
    code: str
    status: Literal["improved"]
    modification: str


def _prepend_summary(prev_result, parts_list):
    s = (prev_result or {}).get("steps_summary")
    if not s:
        return parts_list
    return [f"## Previous steps summary (avoid repeating these mistakes)\n{s}\n"] + parts_list


class TirCodeAgent:
    def __init__(self, config: OpenRouterConfig):
        self.config = config
        self._cfg = load_prompt_config("tir_code")
        syntax = load_tir_syntax()
        main = self._cfg.get("system", "").strip()
        self._system = f"{syntax}\n\n{main}" if syntax else main
        self._response_format = self._cfg.get("response_format", {})

    def build_user_prompt(self, tir_script: str, prev_result: dict | None, recommendations: str | None = None) -> str:
        if prev_result is None:
            parts = [f"Optimize the following TVM TIR / TVMScript code:\n\n{tir_script}"]
            if recommendations:
                parts.append(f"\n--- Analysis recommendations (follow these) ---\n{recommendations}")
            parts.append('\nReturn ONLY a JSON object with fields "code", "status", "modification" as described above.')
            return "\n".join(_prepend_summary(prev_result, parts))

        failed_code = prev_result.get("failed_code")
        best_code = prev_result.get("best_code")
        if prev_result.get("error") and failed_code:
            parts = _prepend_summary(prev_result, [
                "--- Previous attempt FAILED ---",
                f"Error: {prev_result['error']}",
                "",
                "The following code failed verification. Fix it and return a corrected version. Do NOT return it unchanged.",
                "",
                failed_code,
            ])
        else:
            base = f"Optimize the following TVM TIR / TVMScript code:\n\n{tir_script}"
            parts = _prepend_summary(prev_result, [base, ""])
            if prev_result.get("error"):
                parts.append(f"--- Previous attempt FAILED ---\nError: {prev_result['error']}\nFix the error and produce a correct, optimized version.")
            else:
                lat = prev_result.get("latency_ms")
                profile = prev_result.get("profile_text") or ""
                parts.append(f"--- Previous attempt succeeded ---\nLatency: {lat:.3f} ms")
                if profile:
                    parts.append(f"Profile:\n{profile}")
                parts.append("Try to further improve performance.")

        if best_code:
            best_lat = prev_result.get("best_latency_ms")
            best_profile = prev_result.get("best_profile_text") or ""
            parts.append(f"\n--- Best attempt so far (latency: {best_lat:.3f} ms) ---\n{best_code}")
            if best_profile:
                parts.append(f"Its profile:\n{best_profile}")

        if recommendations:
            parts.append(f"\n--- Analysis recommendations (follow these) ---\n{recommendations}")
        parts.append('\nReturn ONLY a JSON object with fields "code", "status", "modification" as described above.')
        return "\n".join(parts)

    def call(self, user_prompt: str) -> dict:
        messages = [
            OpenRouterMessage(role="system", content=self._system),
            OpenRouterMessage(role="user", content=user_prompt),
        ]
        response = call_openrouter(
            self.config,
            messages,
            response_format=self._response_format,
        )
        if "choices" not in response or not response["choices"]:
            raise RuntimeError(f"Invalid response from OpenRouter: {response}")
        content = response["choices"][0]["message"]["content"]
        url = f"{self.config.base_url}/chat/completions"
        return {
            "transformed_content": content,
            "system_prompt": self._system,
            "user_prompt": user_prompt,
            "llm_response": response,
            "request_data": {"messages": [m.model_dump() for m in messages]},
            "model": self.config.model,
            "url": url,
        }


def parse_tir_response(raw: str) -> tuple[TIRTransformResponse | None, str | None]:
    """Returns (parsed, error_msg). error_msg is None on success."""
    parsed = TIRTransformResponse.model_validate_json(raw)
    if not parsed.code.strip():
        return None, "LLM structured output (TIR) has empty 'code' field"
    return parsed, None
