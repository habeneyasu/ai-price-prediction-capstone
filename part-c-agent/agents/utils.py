"""Utility functions for agent implementations."""

from typing import Optional
from shared.price_prediction_utils.frontier_models import create_model, ModelProvider
from shared.price_prediction_utils.settings import settings


def create_agent_model(
    provider: str = "openrouter",
    model_name: Optional[str] = None,
) -> object:
    """
    Create a model instance for agent use.

    Args:
        provider: Model provider ("openrouter" or "ollama")
        model_name: Specific model name (uses default from settings if None)

    Returns:
        Model instance
    """
    return create_model(
        provider=ModelProvider(provider),
        model_name=model_name or settings.model.model_name,
        api_key=settings.openrouter_api_key,
        base_url=(
            settings.openrouter_base_url
            if provider == "openrouter"
            else settings.ollama_base_url
        ),
    )
