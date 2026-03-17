from abc import ABC, abstractmethod

PROMPT_REGISTRY = {}

class BasePromptStrategy(ABC):
    @abstractmethod
    def generate(self, op) -> str:
        pass

def register_prompt(language: str, strategy_name: str):
    def decorator(cls):
        PROMPT_REGISTRY.setdefault(language, {})[strategy_name] = cls()
        return cls
    return decorator
