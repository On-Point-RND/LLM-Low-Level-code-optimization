from ..schemas import OpenRouterConfig, OpenRouterMessage
from .base import load_prompt, call_openrouter


class TirSummaryAgent:
    def __init__(self, config: OpenRouterConfig):
        self.config = config
        self._system = load_prompt("tir_summary")

    def build_user_message(
        self, step_n: int, prev_steps_summary: str, recommendations: str | None,
        code: str, correctness: dict | None, profile_text: str | None,
        error: str | None, latency_ms: float | None,
    ) -> str:
        parts = []
        if prev_steps_summary:
            parts.extend(["## Previous steps summary\n", prev_steps_summary, ""])
        parts.append(f"## Step {step_n}")
        if recommendations:
            parts.append(f"Recommendations for this step:\n{recommendations[:1500]}{'...' if len(recommendations) > 1500 else ''}")
        parts.append(f"Code produced:\n{code[:2000]}{'...' if len(code) > 2000 else ''}")
        if error:
            parts.append(f"Outcome: ERROR\n{error}")
        else:
            ok = (correctness or {}).get("is_correct", False)
            parts.append(f"Outcome: {'SUCCESS' if ok else 'FAILED (correctness)'}")
            if ok and latency_ms is not None:
                parts.append(f"Latency: {latency_ms:.3f} ms")
            if profile_text:
                if len(profile_text) < 800:
                    parts.append(f"Profile:\n{profile_text}")
                else:
                    parts.append(f"Profile (excerpt):\n{profile_text[:800]}...")
        return "\n\n".join(parts)

    def call(self, user_message: str) -> str:
        messages = [
            OpenRouterMessage(role="system", content=self._system),
            OpenRouterMessage(role="user", content=user_message),
        ]
        response = call_openrouter(self.config, messages)
        if "choices" not in response or not response["choices"]:
            raise RuntimeError(f"Invalid response from OpenRouter: {response}")
        return response["choices"][0]["message"]["content"].strip()
