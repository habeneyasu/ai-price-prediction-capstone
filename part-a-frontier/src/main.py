#!/usr/bin/env python3
"""
Part A: Frontier Models - Main Entry Point

Project 6: Capstone Part A – Predict product prices from short descriptions 
using Frontier models (OpenRouter and Ollama).
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.price_prediction_utils import (
    create_model,
    PricePredictor,
    evaluate,
    setup_logging,
    settings,
)
from shared.price_prediction_utils.data_loader import (
    load_from_csv,
    load_from_json,
    load_from_huggingface,
    create_sample_items,
)
from shared.price_prediction_utils.item import Item

# Setup logging
setup_logging(level=settings.log_level)
logger = logging.getLogger(__name__)


def load_data(source: str, **kwargs) -> list:
    """
    Load data from various sources.
    
    Args:
        source: Data source identifier or file path
        **kwargs: Additional arguments for data loading
        
    Returns:
        List of Item objects
    """
    source_lower = source.lower()
    
    if source_lower.startswith("hf:"):
        # HuggingFace dataset
        dataset_name = source[3:]  # Remove "hf:" prefix
        split = kwargs.get("split", "test")
        hf_token = kwargs.get("hf_token")
        return load_from_huggingface(dataset_name, split=split, hf_token=hf_token)
    
    elif source_lower.endswith(".csv"):
        # CSV file - only pass valid CSV arguments
        valid_csv_args = ["title_col", "category_col", "price_col", "description_col", "summary_col"]
        csv_kwargs = {k: v for k, v in kwargs.items() if k in valid_csv_args}
        return load_from_csv(source, **csv_kwargs)
    
    elif source_lower.endswith(".json"):
        # JSON file
        return load_from_json(source)
    
    elif source_lower == "sample":
        # Sample data
        count = kwargs.get("count", 10)
        return create_sample_items(count)
    
    else:
        raise ValueError(f"Unknown data source: {source}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Part A: Predict product prices using Frontier models (OpenRouter/Ollama)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick prediction
  python main.py --provider openrouter --predict "Wireless headphones with noise cancellation"

  # Evaluate with OpenRouter on sample data
  python main.py --provider openrouter --model openai/gpt-4o --data sample --count 20

  # Evaluate with Ollama locally
  python main.py --provider ollama --model llama3.1 --data test_data.csv

  # Evaluate on HuggingFace dataset
  python main.py --provider openrouter --data hf:username/dataset_name --split test
        """,
    )
    
    # Model configuration
    parser.add_argument(
        "--provider",
        type=str,
        default=settings.model.provider,
        choices=["openrouter", "ollama"],
        help="Model provider (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.model.model_name or "openai/gpt-4o-mini",
        help="Specific model name (default: gpt-4o-mini for cost efficiency)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=settings.model.temperature,
        help="Sampling temperature (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=settings.model.max_tokens,
        help="Maximum tokens in response (default: %(default)s)",
    )
    
    # Data configuration
    parser.add_argument(
        "--data",
        type=str,
        help="Data source: CSV/JSON file path, 'hf:dataset_name', or 'sample'",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of samples when using 'sample' data source (default: %(default)s)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split for HuggingFace (default: %(default)s)",
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--max-samples",
        type=int,
        default=settings.evaluation.max_samples or 20,
        help="Maximum number of samples to evaluate (default: 20 for cost efficiency, 0 = all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.evaluation.num_workers,
        help="Number of parallel workers (default: %(default)s)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation",
    )
    
    # Quick prediction mode
    parser.add_argument(
        "--predict",
        type=str,
        help="Quick prediction mode: predict price for a single description",
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save evaluation report (JSON format)",
    )
    
    args = parser.parse_args()
    
    try:
        # Create model
        logger.info(f"Initializing {args.provider} model...")
        
        model_kwargs = {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        
        if args.provider == "openrouter":
            model_kwargs["api_key"] = settings.openrouter_api_key
            model_kwargs["base_url"] = settings.openrouter_base_url or "https://openrouter.ai/api/v1"
        elif args.provider == "ollama":
            model_kwargs["base_url"] = settings.ollama_base_url or "http://localhost:11434"
        
        model = create_model(
            provider=args.provider,
            model_name=args.model,
            **model_kwargs
        )
        
        predictor = PricePredictor(model)
        
        # Quick prediction mode
        if args.predict:
            item = Item(
                title="Custom Item",
                category="Unknown",
                price=0.0,
                description=args.predict,
            )
            predicted_price = predictor.predict(item)
            print(f"\nPredicted Price: ${predicted_price:.2f}\n")
            return
        
        # Load data
        if not args.data:
            logger.error("--data argument is required (use --predict for quick prediction)")
            sys.exit(1)
        
        logger.info(f"Loading data from: {args.data}")
        test_data = load_data(
            args.data,
            count=args.count,
            split=args.split,
        )
        
        if not test_data:
            logger.error("No data loaded!")
            sys.exit(1)
        
        logger.info(f"Loaded {len(test_data)} items")
        
        # Create predictor function
        def predict_func(item):
            return predictor.predict(item)
        
        # Run evaluation
        max_samples = args.max_samples if args.max_samples > 0 else None
        metrics = evaluate(
            predictor=predict_func,
            test_data=test_data,
            name=f"{args.provider} - {args.model or 'default'}",
            max_samples=max_samples,
            num_workers=args.workers,
            plot=not args.no_plot,
        )
        
        # Print model stats
        stats = predictor.get_stats()
        logger.info(f"\nModel Statistics:")
        logger.info(f"  Predictions made: {stats['predictions_made']}")
        logger.info(f"  Errors: {stats['errors']}")
        logger.info(f"  Success rate: {stats['success_rate']:.2%}")
        
        # Save report if requested
        if args.output:
            from shared.price_prediction_utils.evaluator import PriceEvaluator
            evaluator = PriceEvaluator(
                predictor=predict_func,
                test_data=test_data[:max_samples] if max_samples else test_data,
                name=f"{args.provider} - {args.model or 'default'}",
            )
            evaluator.evaluate(verbose=False)
            report = evaluator.generate_report(save_path=args.output)
            logger.info(f"Report saved to {args.output}")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
