"""Provider-specific model construction, selected by node name."""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from customer_service.config.settings import ModelProvider, Settings


def create_chat_model(node_name: str, settings: Settings) -> ChatGoogleGenerativeAI:
    config = settings.model_for(node_name)
    if config.provider is ModelProvider.GOOGLE_GENAI:
        return ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=settings.require_secret("GOOGLE_API_KEY", settings.google_api_key),
            temperature=config.temperature,
            **({"thinking_budget": config.thinking_budget} if config.thinking_budget is not None else {}),
        )
    raise ValueError(f"Unsupported model provider: {config.provider}")
