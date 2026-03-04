import json
import os
import urllib.request
import yaml

from ..schemas import OpenRouterConfig, OpenRouterRequest, OpenRouterMessage


def build_agent_config(full_config: dict, agent_name: str) -> OpenRouterConfig:
    """Build OpenRouterConfig for agent from config.agents.<agent_name>."""
    agent_dict = (full_config.get("agents") or {}).get(agent_name)
    if not agent_dict:
        raise ValueError(f"config.agents.{agent_name} is required")
    return OpenRouterConfig(**agent_dict)


def _prompts_dir():
    return os.path.join(os.path.dirname(__file__), "..", "..", "prompts")


def load_prompt(name: str) -> str:
    """Load prompt from prompts/{name}.yaml. Returns 'system' field."""
    path = os.path.join(_prompts_dir(), f"{name}.yaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("system", "").strip()


def load_tir_syntax() -> str:
    """Load base TIR syntax reference from prompts/tir_syntax.yaml."""
    path = os.path.join(_prompts_dir(), "tir_syntax.yaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return (data.get("syntax") or data.get("system") or "").strip()


def load_prompt_config(name: str) -> dict:
    """Load full prompt config from prompts/{name}.yaml."""
    path = os.path.join(_prompts_dir(), f"{name}.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def call_openrouter(
    config: OpenRouterConfig,
    messages: list,
    response_format: dict | None = None,
) -> dict:
    """Call OpenRouter API. Returns raw response dict with choices."""
    openrouter_req = OpenRouterRequest(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        top_k=config.top_k,
        response_format=response_format,
    )
    url = f"{config.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    request_data = openrouter_req.model_dump(exclude_none=True)
    data = json.dumps(request_data).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=config.timeout) as response:
        return json.loads(response.read().decode("utf-8"))
