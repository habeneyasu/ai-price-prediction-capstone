"""Messaging agent for crafting and sending deal notifications."""

from typing import Optional
from .agent import Agent
from .deals import Opportunity
from .utils import create_agent_model
from ..notifier.notifier import Notifier


class MessagingAgent(Agent):
    """Agent responsible for crafting and sending deal notifications."""

    name = "Messaging Agent"

    def __init__(self, model_provider: str = "openrouter", model_name: str = None):
        """
        Initialize messaging agent.

        Args:
            model_provider: Model provider for message crafting
            model_name: Specific model name
        """
        super().__init__()
        self.log("Initializing Messaging Agent")
        self.model = create_agent_model(provider=model_provider, model_name=model_name)
        self.notifier = Notifier()
        self.log("Messaging Agent ready")

    def craft_message(
        self, description: str, deal_price: float, estimated_value: float
    ) -> str:
        """
        Craft an engaging notification message using LLM.

        Args:
            description: Product description
            deal_price: Deal price
            estimated_value: Estimated true value

        Returns:
            Crafted notification message
        """
        prompt = (
            "Please summarize this great deal in 2-3 sentences to be sent as an exciting "
            "push notification alerting the user about this deal.\n\n"
            f"Item Description: {description}\n"
            f"Offered Price: ${deal_price:.2f}\n"
            f"Estimated true value: ${estimated_value:.2f}\n\n"
            "Respond only with the 2-3 sentence message which will be used to alert & "
            "excite the user about this deal."
        )

        try:
            response = self.model.predict_price(prompt)
            message = str(response).strip()
            if len(message) > 200:
                message = message[:197] + "..."
            return message
        except Exception as e:
            self.log(f"Error crafting message: {e}, using default", "WARNING")
            return (
                f"Deal Alert! Price=${deal_price:.2f}, "
                f"Estimate=${estimated_value:.2f}, "
                f"Discount=${estimated_value - deal_price:.2f}. "
                f"{description[:50]}..."
            )

    def alert(self, opportunity: Opportunity) -> None:
        """
        Send alert notification for an opportunity.

        Args:
            opportunity: Opportunity to alert about
        """
        self.log("Crafting notification message")
        message = self.craft_message(
            opportunity.deal.product_description,
            opportunity.deal.price,
            opportunity.estimate,
        )

        full_message = f"{message}\n\nURL: {opportunity.deal.url}"
        self.notifier.send(full_message, opportunity)
        self.log("Notification sent")
