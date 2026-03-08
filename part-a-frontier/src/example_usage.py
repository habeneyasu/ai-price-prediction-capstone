#!/usr/bin/env python3
"""
Example usage of Part A: Frontier Models for price prediction.

This script demonstrates how to use the price prediction system programmatically.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.price_prediction_utils import (
    create_model,
    PricePredictor,
    evaluate,
    setup_logging,
    settings,
    Item,
    create_sample_items,
)

# Setup logging
setup_logging(level="INFO")


def example_single_prediction():
    """Example 1: Predict price for a single item."""
    print("Example 1: Single Item Prediction")
    print("-" * 60)
    
    # Create model (OpenRouter)
    try:
        model = create_model(
            provider="openrouter",
            model_name="openai/gpt-4o",
            temperature=0.3,
            api_key=settings.openrouter_api_key,
        )
        predictor = PricePredictor(model)
        
        # Create a test item
        item = Item(
            title="Wireless Bluetooth Headphones",
            category="Electronics",
            price=0.0,  # Unknown - we're predicting this
            summary="High-quality wireless headphones with active noise cancellation, "
                    "30-hour battery life, and premium sound quality. Includes "
                    "carrying case and USB-C charging cable."
        )
        
        predicted_price = predictor.predict(item)
        print(f"Item: {item.title}")
        print(f"Description: {item.summary}")
        print(f"Predicted Price: ${predicted_price:.2f}")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Make sure you have OPENROUTER_API_KEY set in your .env file")
        print()


def example_batch_evaluation():
    """Example 2: Evaluate on sample data."""
    print("\nExample 2: Batch Evaluation")
    print("-" * 60)
    
    # Create sample items
    sample_items = create_sample_items(count=5)
    
    try:
        # Create model
        model = create_model(
            provider="openrouter",
            model_name="openai/gpt-4o",
            temperature=0.3,
            api_key=settings.openrouter_api_key,
        )
        predictor = PricePredictor(model)
        
        # Evaluate
        metrics = evaluate(
            predictor=predictor.predict,
            test_data=sample_items,
            name="GPT-4o Example",
            max_samples=5,
            num_workers=2,
            plot=False,  # Set to True to see plots
        )
        
        print(f"\nEvaluation Results:")
        print(f"  MAE: ${metrics['mae']:.2f}")
        print(f"  RMSE: ${metrics['rmse']:.2f}")
        print(f"  R²: {metrics['r2_percentage']:.2f}%")
        print()
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print()


def example_ollama():
    """Example 3: Using Ollama for local inference."""
    print("\nExample 3: Local Inference with Ollama")
    print("-" * 60)
    
    try:
        # Create Ollama model
        model = create_model(
            provider="ollama",
            model_name="llama3.1",
            temperature=0.3,
            base_url=settings.ollama_base_url or "http://localhost:11434",
        )
        
        # Check availability
        if hasattr(model, 'check_availability'):
            if not model.check_availability():
                print("Ollama model not available.")
                print("Make sure Ollama is running and model is pulled:")
                print("  ollama pull llama3.1")
                return
        
        predictor = PricePredictor(model)
        
        # Test prediction
        item = create_sample_items(count=1)[0]
        predicted_price = predictor.predict(item)
        
        print(f"Item: {item.title}")
        print(f"Actual Price: ${item.price:.2f}")
        print(f"Predicted Price: ${predicted_price:.2f}")
        print(f"Error: ${abs(predicted_price - item.price):.2f}")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running: https://ollama.ai")
        print()


def example_model_comparison():
    """Example 4: Compare different models."""
    print("\nExample 4: Model Comparison")
    print("-" * 60)
    
    test_item = create_sample_items(count=1)[0]
    
    models_to_try = [
        ("openrouter", "openai/gpt-4o", "OpenRouter GPT-4o"),
        ("openrouter", "anthropic/claude-3.5-sonnet", "OpenRouter Claude 3.5"),
    ]
    
    results = []
    
    for provider, model_name, display_name in models_to_try:
        try:
            if provider == "openrouter":
                model = create_model(
                    provider=provider,
                    model_name=model_name,
                    api_key=settings.openrouter_api_key,
                )
            else:
                model = create_model(provider=provider, model_name=model_name)
            
            predictor = PricePredictor(model)
            predicted = predictor.predict(test_item)
            error = abs(predicted - test_item.price)
            
            results.append({
                "Model": display_name,
                "Predicted": f"${predicted:.2f}",
                "Actual": f"${test_item.price:.2f}",
                "Error": f"${error:.2f}",
            })
            
        except Exception as e:
            print(f"Failed to test {display_name}: {e}")
    
    if results:
        print(f"\nTest Item: {test_item.title}")
        print(f"Description: {test_item.summary[:80]}...")
        print("\nResults:")
        for result in results:
            print(f"  {result['Model']}: {result['Predicted']} (Error: {result['Error']})")
        print()


def main():
    """Run all examples."""
    print("Part A: Frontier Models - Example Usage")
    print("=" * 60)
    
    # Run examples
    example_single_prediction()
    example_batch_evaluation()
    example_ollama()
    example_model_comparison()
    
    print("\nExamples completed.")


if __name__ == "__main__":
    main()
