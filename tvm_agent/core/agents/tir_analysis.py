from ..schemas import OpenRouterConfig, OpenRouterMessage
from .base import load_prompt, load_tir_syntax, call_openrouter


class TirAnalysisAgent:
    def __init__(self, config: OpenRouterConfig):
        self.config = config
        syntax = load_tir_syntax()
        main = load_prompt("tir_analysis")
        self._system = f"{syntax}\n\n{main}" if syntax else main

    def _ref_section(self, reference_python_code: str) -> str:
        if not reference_python_code.strip():
            return ""
        return (
            "## Reference (context only — do NOT output Python, output TIR recommendations)\n"
            "Original PyTorch/Python code that the TIR implements:\n\n```python\n"
            + reference_python_code.strip() + "\n```\n\n"
        )

    def build_user_message(self, tir_script: str, prev_result: dict | None, reference_python_code: str = "") -> str:
        ref = self._ref_section(reference_python_code)
        if prev_result is None:
            return "\n".join([
                ref,
                "## Current TIR code\n", tir_script,
                "\n## Feedback\nInitial code, no previous attempt. Analyze and suggest the first optimization step.",
                "\n\nWrite your recommendations:",
            ])
        failed_code = prev_result.get("failed_code")
        code = failed_code if (prev_result.get("error") and failed_code) else tir_script
        parts = []
        if ref:
            parts.append(ref)
        if prev_result.get("steps_summary"):
            parts.append(f"## Previous steps summary (avoid repeating these mistakes)\n{prev_result['steps_summary']}\n")
        parts.extend(["## Current TIR code\n", code])
        if prev_result.get("error"):
            parts.extend(["\n## Feedback (previous attempt failed)\n", prev_result["error"]])
        else:
            lat = prev_result.get("latency_ms")
            profile = prev_result.get("profile_text") or ""
            parts.append(f"\n## Feedback (previous attempt succeeded, latency: {lat:.3f} ms)\n")
            if profile:
                parts.append(profile)
        if prev_result.get("best_code"):
            best_lat = prev_result.get("best_latency_ms")
            parts.append(f"\n## Best attempt so far (latency: {best_lat:.3f} ms)\n")
            parts.append(prev_result["best_code"])
        parts.append("\n\nWrite your recommendations:")
        return "\n".join(parts)

    def call(self, user_message: str) -> str:
        messages = [
            OpenRouterMessage(role="system", content=self._system),
            OpenRouterMessage(role="user", content=user_message),
        ]
        response = call_openrouter(self.config, messages)
        if "choices" not in response or not response["choices"]:
            raise RuntimeError(f"Invalid response from OpenRouter: {response}")
        return response["choices"][0]["message"]["content"].strip()
