from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any

class OpenRouterMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(..., description="Роль сообщения")
    content: str = Field(..., description="Содержимое сообщения")

class OpenRouterRequest(BaseModel):
    model: str = Field(..., description="ID модели OpenRouter (например, 'openai/gpt-4', 'anthropic/claude-3-opus')")
    messages: List[OpenRouterMessage] = Field(..., description="Список сообщений для контекста")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Температура генерации")
    max_tokens: Optional[int] = Field(default=None, gt=0, description="Максимальное количество токенов в ответе")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: Optional[int] = Field(default=None, gt=0, description="Top-k sampling (vLLM etc.)")
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0, description="Штраф за частоту")
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0, description="Штраф за присутствие")
    stop: Optional[List[str]] = Field(default=None, description="Стоп-последовательности")
    stream: bool = Field(default=False, description="Потоковая генерация")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Метаданные запроса")
    # Strict structured output (OpenRouter `response_format`)
    response_format: Optional[Dict[str, Any]] = Field(
        default=None, description="Формат структурированного ответа (json_schema и т.п.)"
    )

class OpenRouterChoice(BaseModel):
    index: int = Field(..., description="Индекс варианта ответа")
    message: OpenRouterMessage = Field(..., description="Сообщение от модели")
    finish_reason: Optional[str] = Field(default=None, description="Причина завершения (stop, length, content_filter)")

class OpenRouterUsage(BaseModel):
    prompt_tokens: int = Field(..., description="Количество токенов в промпте")
    completion_tokens: int = Field(..., description="Количество токенов в ответе")
    total_tokens: int = Field(..., description="Общее количество токенов")

class OpenRouterResponse(BaseModel):
    id: str = Field(..., description="ID запроса")
    model: str = Field(..., description="Использованная модель")
    choices: List[OpenRouterChoice] = Field(..., description="Варианты ответа")
    usage: Optional[OpenRouterUsage] = Field(default=None, description="Статистика использования")
    created: int = Field(..., description="Время создания ответа (Unix timestamp)")

class OpenRouterConfig(BaseModel):
    api_key: str = Field(..., description="API ключ")
    base_url: str = Field(default="https://openrouter.ai/api/v1", description="Base URL API")
    model: str = Field(default="openai/gpt-4o-mini", description="ID модели")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, gt=0)
    timeout: float = Field(default=60.0, gt=0, description="Таймаут запроса в секундах")

class LLMTransformRequest(BaseModel):
    content: str = Field(..., description="Исходный контент (TVMScript или JSON граф)")
    format: Literal["python", "json"] = Field(..., description="Формат контента")
    task: str = Field(default="optimize", description="Задача трансформации")
    model: Optional[str] = Field(default=None, description="Модель для использования (переопределяет default_model)")

class LLMTransformResponse(BaseModel):
    transformed_content: str = Field(..., description="Трансформированный контент")
    format: Literal["python", "json"] = Field(..., description="Формат результата")
    success: bool = Field(default=True, description="Успешность трансформации")
    error: Optional[str] = Field(default=None, description="Сообщение об ошибке, если есть")
    usage: Optional[OpenRouterUsage] = Field(default=None, description="Статистика использования токенов")
