#!/usr/bin/env python3
"""
Cost Optimization Guide and Utilities for Part A.

This module provides cost-saving strategies and utilities for price prediction
using frontier models.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.price_prediction_utils import create_model, PricePredictor, Item, create_sample_items

def estimate_cost(num_predictions: int, model_name: str = "openai/gpt-4o-mini") -> dict:
    """
    Estimate API costs for predictions.
    
    Args:
        num_predictions: Number of predictions to make
        model_name: Model to use
        
    Returns:
        Dictionary with cost estimates
    """
    # Approximate costs per 1K tokens (as of 2024)
    # These are estimates - check OpenRouter pricing for current rates
    cost_per_1k_tokens = {
        "openai/gpt-4o": 0.005,  # $5 per 1M input tokens
        "openai/gpt-4o-mini": 0.00015,  # $0.15 per 1M input tokens
        "anthropic/claude-3-haiku": 0.00025,  # $0.25 per 1M input tokens
        "meta-llama/llama-3.1-8b-instruct": 0.0001,  # Very cheap
    }
    
    # Average tokens per prediction (system prompt + user prompt + response)
    avg_tokens_per_prediction = 200  # Conservative estimate
    
    model_cost = cost_per_1k_tokens.get(model_name, 0.00015)
    total_tokens = num_predictions * avg_tokens_per_prediction
    estimated_cost = (total_tokens / 1000) * model_cost
    
    return {
        "num_predictions": num_predictions,
        "model": model_name,
        "estimated_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost,
        "cost_per_prediction": estimated_cost / num_predictions if num_predictions > 0 else 0,
    }


def print_cost_optimization_tips():
    """Print cost optimization tips."""
    print("=" * 70)
    print("COST OPTIMIZATION TIPS")
    print("=" * 70)
    print()
    print("1. USE CHEAPER MODELS:")
    print("   - gpt-4o-mini: ~$0.15 per 1M tokens (RECOMMENDED)")
    print("   - claude-3-haiku: ~$0.25 per 1M tokens")
    print("   - llama-3.1-8b: ~$0.10 per 1M tokens (cheapest)")
    print("   - gpt-4o: ~$5 per 1M tokens (expensive, use only for final evaluation)")
    print()
    print("2. LIMIT EVALUATION SIZE:")
    print("   - Use --max-samples 20 for quick testing")
    print("   - Use --max-samples 50 for development")
    print("   - Use --max-samples 200 for final evaluation")
    print()
    print("3. USE OLLAMA FOR FREE LOCAL INFERENCE:")
    print("   - Install Ollama: https://ollama.ai")
    print("   - Pull model: ollama pull llama3.1")
    print("   - Use: --provider ollama --model llama3.1")
    print()
    print("4. OPTIMIZE PROMPTS:")
    print("   - Keep descriptions concise")
    print("   - Use --max-tokens 20 (default is 50)")
    print()
    print("5. CACHE PREDICTIONS:")
    print("   - Save results to avoid re-predicting")
    print("   - Use CSV/JSON files for reproducible evaluations")
    print()
    print("=" * 70)
    print()


def compare_model_costs(num_predictions: int = 100):
    """Compare costs across different models."""
    print(f"\nCost Comparison for {num_predictions} Predictions:\n")
    
    models = [
        "openai/gpt-4o-mini",
        "anthropic/claude-3-haiku",
        "meta-llama/llama-3.1-8b-instruct",
        "openai/gpt-4o",
    ]
    
    print(f"{'Model':<40} {'Estimated Cost':<20}")
    print("-" * 60)
    
    for model in models:
        cost_info = estimate_cost(num_predictions, model)
        print(f"{model:<40} ${cost_info['estimated_cost_usd']:.4f}")
    
    print("\nRecommendation: Use gpt-4o-mini for best balance of cost and quality")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cost optimization utilities")
    parser.add_argument(
        "--predictions",
        type=int,
        default=100,
        help="Number of predictions to estimate cost for"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model to estimate costs for"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare costs across different models"
    )
    parser.add_argument(
        "--tips",
        action="store_true",
        help="Show cost optimization tips"
    )
    
    args = parser.parse_args()
    
    if args.tips:
        print_cost_optimization_tips()
    
    if args.compare:
        compare_model_costs(args.predictions)
    else:
        cost_info = estimate_cost(args.predictions, args.model)
        print(f"\nCost Estimate for {args.predictions} predictions using {args.model}:")
        print(f"   Estimated tokens: {cost_info['estimated_tokens']:,}")
        print(f"   Estimated cost: ${cost_info['estimated_cost_usd']:.4f}")
        print(f"   Cost per prediction: ${cost_info['cost_per_prediction']:.6f}")
        print()
