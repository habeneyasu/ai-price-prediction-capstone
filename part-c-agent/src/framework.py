"""Main orchestration framework for autonomous deal detection."""

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports when running as script
if __name__ == "__main__" or not __package__:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.planning_agent import PlanningAgent
from agents.deals import Opportunity

logger = logging.getLogger(__name__)


class DealAgentFramework:
    """Main framework for orchestrating autonomous deal detection."""

    MEMORY_FILENAME = "memory.json"

    def __init__(
        self,
        deal_threshold: float = 50.0,
        scanner_provider: str = "openrouter",
        ensemble_primary: str = "openrouter",
        memory_file: str = None,
    ):
        """
        Initialize the framework.

        Args:
            deal_threshold: Minimum discount threshold
            scanner_provider: Provider for scanner agent
            ensemble_primary: Primary provider for ensemble agent
            memory_file: Path to memory file
        """
        self.memory_file = Path(memory_file or self.MEMORY_FILENAME)
        self.memory = self.read_memory()
        self.planner = PlanningAgent(
            deal_threshold=deal_threshold,
            scanner_provider=scanner_provider,
            ensemble_primary=ensemble_primary,
        )
        logger.info("Deal Agent Framework initialized")

    def read_memory(self) -> List[Opportunity]:
        """Read opportunities from memory file."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r") as f:
                    data = json.load(f)
                return [Opportunity(**item) for item in data]
            except Exception as e:
                logger.warning(f"Error reading memory: {e}")
                return []
        return []

    def write_memory(self) -> None:
        """Write opportunities to memory file."""
        try:
            data = [opp.model_dump() for opp in self.memory]
            with open(self.memory_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing memory: {e}")

    def get_memory_urls(self) -> List[str]:
        """Get list of URLs from memory."""
        return [opp.deal.url for opp in self.memory]

    def run(self) -> Optional[Opportunity]:
        """
        Run one cycle of deal detection.

        Returns:
            New opportunity if found, None otherwise
        """
        logger.info("Starting deal detection cycle")
        memory_urls = self.get_memory_urls()
        result = self.planner.plan(memory=memory_urls)

        if result:
            self.memory.append(result)
            self.write_memory()
            logger.info(f"New opportunity found: ${result.discount:.2f} discount")
        else:
            logger.info("No new opportunities found")

        return result

    def get_opportunities(self) -> List[Opportunity]:
        """Get all opportunities from memory."""
        return self.memory
