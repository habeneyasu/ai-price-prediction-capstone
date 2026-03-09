"""Planning agent that orchestrates the deal detection workflow."""

from typing import List, Optional
from .agent import Agent
from .deals import Deal, Opportunity
from .scanner_agent import ScannerAgent
from .ensemble_agent import EnsembleAgent
from .messaging_agent import MessagingAgent


class PlanningAgent(Agent):
    """Agent that orchestrates the deal detection and notification workflow."""

    name = "Planning Agent"
    DEAL_THRESHOLD = 50.0

    def __init__(
        self,
        deal_threshold: float = 50.0,
        scanner_provider: str = "openrouter",
        ensemble_primary: str = "openrouter",
    ):
        """
        Initialize planning agent with sub-agents.

        Args:
            deal_threshold: Minimum discount threshold for notifications
            scanner_provider: Provider for scanner agent
            ensemble_primary: Primary provider for ensemble agent
        """
        super().__init__()
        self.log("Initializing Planning Agent")
        self.DEAL_THRESHOLD = deal_threshold
        self.scanner = ScannerAgent(model_provider=scanner_provider)
        self.ensemble = EnsembleAgent(primary_provider=ensemble_primary)
        self.messenger = MessagingAgent()
        self.log("Planning Agent ready")

    def run(self, deal: Deal) -> Opportunity:
        """
        Process a single deal through the workflow.

        Args:
            deal: Deal to process

        Returns:
            Opportunity with estimate and discount
        """
        self.log(f"Processing deal: {deal.product_description[:50]}...")
        estimate = self.ensemble.price(deal.product_description)
        discount = estimate - deal.price
        self.log(f"Deal processed - Discount: ${discount:.2f}")
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: List[str] = None) -> Optional[Opportunity]:
        """
        Execute full workflow: scan, evaluate, and notify.

        Args:
            memory: List of URLs already processed

        Returns:
            Best opportunity if above threshold, None otherwise
        """
        if memory is None:
            memory = []

        self.log("Starting deal detection workflow")
        selection = self.scanner.scan(memory=memory)

        if not selection or not selection.deals:
            self.log("No deals found")
            return None

        opportunities = [self.run(deal) for deal in selection.deals[:5]]
        opportunities.sort(key=lambda opp: opp.discount, reverse=True)
        best = opportunities[0]

        self.log(f"Best deal discount: ${best.discount:.2f}")

        if best.discount > self.DEAL_THRESHOLD:
            self.messenger.alert(best)
            self.log("Deal above threshold - notification sent")
            return best
        else:
            self.log(f"Deal below threshold (${self.DEAL_THRESHOLD})")
            return None
