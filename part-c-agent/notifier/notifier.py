"""Notification system for deal alerts."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from ..agents.deals import Opportunity


class Notifier:
    """Handles sending notifications for deals."""

    def __init__(self, output_dir: str = "notifications"):
        """
        Initialize notifier.

        Args:
            output_dir: Directory to save notifications
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.notification_file = self.output_dir / "notifications.jsonl"

    def send(self, message: str, opportunity: Optional[Opportunity] = None) -> None:
        """
        Send a notification.

        Args:
            message: Notification message
            opportunity: Associated opportunity (optional)
        """
        timestamp = datetime.now().isoformat()
        notification = {
            "timestamp": timestamp,
            "message": message,
        }

        if opportunity:
            notification["opportunity"] = {
                "product_description": opportunity.deal.product_description,
                "price": opportunity.deal.price,
                "estimate": opportunity.estimate,
                "discount": opportunity.discount,
                "url": opportunity.deal.url,
            }

        self._save_to_file(notification)
        self._print_to_console(message, opportunity)

    def _save_to_file(self, notification: dict) -> None:
        """Save notification to JSONL file."""
        try:
            with open(self.notification_file, "a") as f:
                f.write(json.dumps(notification) + "\n")
        except Exception as e:
            print(f"Error saving notification: {e}")

    def _print_to_console(self, message: str, opportunity: Optional[Opportunity]) -> None:
        """Print notification to console."""
        print("\n" + "=" * 80)
        print("DEAL ALERT")
        print("=" * 80)
        print(message)
        if opportunity:
            print(f"\nDiscount: ${opportunity.discount:.2f}")
        print("=" * 80 + "\n")
