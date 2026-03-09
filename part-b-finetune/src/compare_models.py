#!/usr/bin/env python3
"""
Compare fine-tuned model with frontier models.

Evaluates both fine-tuned and frontier models on the same test set
for performance comparison.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from datasets import load_dataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.price_prediction_utils import (
    create_model,
    PricePredictor,
    evaluate,
    setup_logging,
    settings,
    Item,
)
from part_b_finetune.src.evaluate import FineTunedPredictor

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


def compare_models(
    base_model: str,
    adapter_path: str,
    test_dataset_name: str,
    frontier_model_name: str = "openai/gpt-4o-mini",
    max_samples: int = 100,
):
    """
    Compare fine-tuned model with frontier model.
    
    Args:
        base_model: Base model identifier for fine-tuned model
        adapter_path: Path to PEFT adapter
        test_dataset_name: HuggingFace test dataset name
        frontier_model_name: Frontier model to compare against
        max_samples: Maximum samples to evaluate
        
    Returns:
        Dictionary with comparison metrics
    """
    logger.info("Loading test dataset...")
    dataset = load_dataset(test_dataset_name, split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Convert to Items for frontier model
    test_items = []
    for item in dataset:
        # Extract description from text
        text = item.get("text", "")
        if "Price is $" in text:
            description = text.split("Price is $")[0].replace("What does this cost to the nearest dollar?", "").strip()
        else:
            description = text
        
        test_items.append(Item(
            title=item.get("title", ""),
            category=item.get("category", "Unknown"),
            price=item.get("price", 0.0),
            description=description,
        ))
    
    results = {}
    
    # Evaluate fine-tuned model
    logger.info("Evaluating fine-tuned model...")
    finetuned_predictor = FineTunedPredictor(
        base_model=base_model,
        adapter_path=adapter_path,
        use_4bit=True,
    )
    
    def finetuned_predict(item):
        return finetuned_predictor.predict(item.description or item.summary or item.title)
    
    finetuned_metrics = evaluate(
        predictor=finetuned_predict,
        test_data=test_items,
        name="Fine-tuned Model",
        max_samples=max_samples,
        plot=True,
    )
    results["finetuned"] = finetuned_metrics
    
    # Evaluate frontier model
    logger.info(f"Evaluating frontier model: {frontier_model_name}...")
    frontier_model = create_model(
        provider="openrouter",
        model_name=frontier_model_name,
        api_key=settings.openrouter_api_key,
    )
    frontier_predictor = PricePredictor(frontier_model)
    
    frontier_metrics = evaluate(
        predictor=frontier_predictor.predict,
        test_data=test_items,
        name=f"Frontier Model ({frontier_model_name})",
        max_samples=max_samples,
        plot=True,
    )
    results["frontier"] = frontier_metrics
    
    # Print comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"\nFine-tuned Model:")
    print(f"  MAE: ${finetuned_metrics['mae']:.2f}")
    print(f"  RMSE: ${finetuned_metrics['rmse']:.2f}")
    print(f"  R²: {finetuned_metrics['r2_percentage']:.2f}%")
    print(f"\nFrontier Model ({frontier_model_name}):")
    print(f"  MAE: ${frontier_metrics['mae']:.2f}")
    print(f"  RMSE: ${frontier_metrics['rmse']:.2f}")
    print(f"  R²: {frontier_metrics['r2_percentage']:.2f}%")
    print("\n" + "=" * 70)
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare fine-tuned and frontier models")
    
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base model identifier for fine-tuned model",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to PEFT adapter",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        required=True,
        help="HuggingFace test dataset name",
    )
    parser.add_argument(
        "--frontier-model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Frontier model to compare (default: openai/gpt-4o-mini)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum samples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save comparison report (JSON)",
    )
    
    args = parser.parse_args()
    
    try:
        results = compare_models(
            base_model=args.base_model,
            adapter_path=args.adapter,
            test_dataset_name=args.test_dataset,
            frontier_model_name=args.frontier_model,
            max_samples=args.max_samples,
        )
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Comparison report saved to {args.output}")
        
    except KeyboardInterrupt:
        logger.info("Comparison interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
