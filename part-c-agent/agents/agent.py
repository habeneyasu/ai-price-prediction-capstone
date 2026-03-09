"""Base agent class for autonomous deal detection system."""

import logging

logger = logging.getLogger(__name__)


class Agent:
    """Base class for all agents in the autonomous deal detection system."""

    name: str = "Agent"
    color: str = ""

    def log(self, message: str, level: str = "INFO") -> None:
        """
        Log a message with agent identification.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, DEBUG)
        """
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(f"[{self.name}] {message}")
