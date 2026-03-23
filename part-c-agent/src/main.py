"""CLI entry point for Part C autonomous agent system."""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.price_prediction_utils.logging_config import setup_logging

# Import framework - handle both script and module execution
try:
    from .framework import DealAgentFramework
except ImportError:
    # If running as script, use absolute import
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.framework import DealAgentFramework

setup_logging()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Autonomous agent system for deal detection"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Minimum discount threshold for notifications (default: 50.0)",
    )
    parser.add_argument(
        "--scanner-provider",
        choices=["openrouter", "ollama"],
        default="openrouter",
        help="Provider for scanner agent (default: openrouter)",
    )
    parser.add_argument(
        "--ensemble-primary",
        choices=["openrouter", "ollama"],
        default="openrouter",
        help="Primary provider for ensemble agent (default: openrouter)",
    )
    parser.add_argument(
        "--memory-file",
        type=str,
        help="Path to memory file (default: memory.json)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of detection cycles to run (default: 1)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all opportunities from memory",
    )

    args = parser.parse_args()

    framework = DealAgentFramework(
        deal_threshold=args.threshold,
        scanner_provider=args.scanner_provider,
        ensemble_primary=args.ensemble_primary,
        memory_file=args.memory_file,
    )

    if args.list:
        opportunities = framework.get_opportunities()
        if not opportunities:
            print("No opportunities in memory.")
            return

        print(f"\nFound {len(opportunities)} opportunities:\n")
        for i, opp in enumerate(opportunities, 1):
            print(f"{i}. {opp.deal.product_description[:60]}...")
            print(f"   Price: ${opp.deal.price:.2f} | Estimate: ${opp.estimate:.2f} | Discount: ${opp.discount:.2f}")
            print(f"   URL: {opp.deal.url}\n")
        return

    for cycle in range(args.cycles):
        if args.cycles > 1:
            print(f"\n--- Cycle {cycle + 1}/{args.cycles} ---\n")
        result = framework.run()
        if result:
            print(f"\n✓ Opportunity found: ${result.discount:.2f} discount")
        else:
            print("\n✗ No opportunities found above threshold")


if __name__ == "__main__":
    main()
