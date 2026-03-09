"""Ensemble agent that combines multiple price prediction models."""

from typing import Optional
from shared.price_prediction_utils.frontier_models import create_model, ModelProvider
from shared.price_prediction_utils.settings import settings
from .agent import Agent
from .utils import create_agent_model


class EnsembleAgent(Agent):
    """Agent that combines multiple models for price prediction."""

    name = "Ensemble Agent"

    def __init__(
        self,
        primary_provider: str = "openrouter",
        primary_model: str = None,
        secondary_provider: str = "ollama",
        secondary_model: str = "llama3.2",
    ):
        """
        Initialize ensemble agent with multiple models.

        Args:
            primary_provider: Primary model provider
            primary_model: Primary model name
            secondary_provider: Secondary model provider
            secondary_model: Secondary model name
        """
        super().__init__()
        self.log("Initializing Ensemble Agent")

        self.primary_model = create_agent_model(
            provider=primary_provider,
            model_name=primary_model,
        )

        if secondary_provider == "ollama":
            self.secondary_model = create_model(
                provider=ModelProvider("ollama"),
                model_name=secondary_model,
                base_url=settings.ollama_base_url,
            )
        else:
            self.secondary_model = None

        self.log("Ensemble Agent ready")

    def price(self, description: str) -> float:
        """
        Estimate price using ensemble of models.

        Args:
            description: Product description

        Returns:
            Estimated price
        """
        self.log("Running ensemble prediction")

        primary_price = self.primary_model.predict_price(description)

        if self.secondary_model:
            try:
                secondary_price = self.secondary_model.predict_price(description)
                combined = primary_price * 0.8 + secondary_price * 0.2
            except Exception as e:
                self.log(f"Secondary model failed: {e}, using primary only", "WARNING")
                combined = primary_price
        else:
            combined = primary_price

        self.log(f"Ensemble prediction: ${combined:.2f}")
        return combined
